"""
scheduler.py — ModelScheduler worker and LatencyTracker.
Now wired to analytics.py for full structured event logging.
"""

import asyncio
import json
import logging
import time
from typing import Optional

from config import settings
from ollama_client import OllamaClient
from queue_manager import RequestQueue, extract_vram_gb, normalise_model_name
from schemas import LatencyRecord, PendingRequest, RequestStatus
from analytics import analytics

logger = logging.getLogger(__name__)


class LatencyTracker:
    def __init__(self, max_records: int = 1_000):
        self._records: list[LatencyRecord] = []
        self._max = max_records

    def record(self, req: PendingRequest) -> None:
        if req.execution_latency_ms is None:
            return
        exec_ms = req.execution_latency_ms
        tok_s   = (req.tokens_generated / (exec_ms / 1000.0)
                   if req.tokens_generated and exec_ms > 0 else 0.0)

        rec = LatencyRecord(
            request_id        = req.request_id,
            model             = req.model,
            endpoint          = req.endpoint,
            enqueue_wall_time = req.wall_enqueue_time,
            queue_wait_ms     = round(req.queue_wait_ms or 0, 2),
            execution_ms      = round(exec_ms, 2),
            total_ms          = round(req.total_latency_ms or 0, 2),
            tokens_generated  = req.tokens_generated,
            tokens_per_second = round(tok_s, 1),
            status            = req.status.value,
        )
        if len(self._records) >= self._max:
            self._records.pop(0)
        self._records.append(rec)

        logger.info(
            "[LATENCY] req=%-8s model=%-22s "
            "queue_wait=%7.1f ms  exec=%8.1f ms  total=%8.1f ms  "
            "tokens=%4d  tok/s=%6.1f  status=%s",
            rec.request_id[:8], rec.model,
            rec.queue_wait_ms, rec.execution_ms, rec.total_ms,
            rec.tokens_generated, rec.tokens_per_second, rec.status,
        )

        # ── Write to JSONL analytics log ──────────────────────────────────────
        was_cold = analytics.mark_model_seen(req.model, req.execution_start_time or 0)
        analytics.log_request_complete(
            request_id    = req.request_id,
            model         = req.model,
            endpoint      = req.endpoint,
            queue_wait_ms = rec.queue_wait_ms,
            exec_ms       = rec.execution_ms,
            total_ms      = rec.total_ms,
            tokens        = rec.tokens_generated,
            tok_per_s     = rec.tokens_per_second,
            status        = rec.status,
            streaming     = req.streaming,
            was_cold_load = was_cold,
        )

    def get_summary(self, model: Optional[str] = None) -> dict:
        recs = [r for r in self._records if model is None or r.model == model]
        if not recs:
            return {"count": 0}

        def pct(lst, p):
            return round(sorted(lst)[max(0, int(len(lst)*p/100)-1)], 1)

        exec_ms  = [r.execution_ms      for r in recs]
        wait_ms  = [r.queue_wait_ms     for r in recs]
        tok_s    = [r.tokens_per_second for r in recs]
        return {
            "count":             len(recs),
            "avg_exec_ms":       round(sum(exec_ms) / len(exec_ms), 1),
            "p50_exec_ms":       pct(exec_ms, 50),
            "p95_exec_ms":       pct(exec_ms, 95),
            "p99_exec_ms":       pct(exec_ms, 99),
            "avg_queue_wait_ms": round(sum(wait_ms) / len(wait_ms), 1),
            "avg_tok_per_s":     round(sum(tok_s)   / len(tok_s),   1),
        }

    def all_models(self) -> set[str]:
        return {r.model for r in self._records}


latency_tracker = LatencyTracker()


class ModelScheduler:
    def __init__(self, queue: RequestQueue, ollama: OllamaClient):
        self.queue  = queue
        self.ollama = ollama
        self._running     = False
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._running     = True
        self._worker_task = asyncio.create_task(
            self._run_loop(), name="scheduler-worker"
        )
        # Background hourly summary writer
        asyncio.create_task(analytics.run_hourly_summary(), name="hourly-summary")
        logger.info("🚀  ModelScheduler worker started")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self.ollama.close()
        analytics.close()
        logger.info("🛑  ModelScheduler stopped cleanly")

    async def _run_loop(self) -> None:
        logger.info("[SCHEDULER] Event loop running")
        while self._running:
            try:
                await asyncio.wait_for(self.queue.wait_for_items(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            await asyncio.sleep(settings.BATCH_COLLECT_TIMEOUT_S)
            batch = await self.queue.get_batch()
            if not batch:
                continue

            # ── Log batch formation ───────────────────────────────────────────
            analytics.log_batch_formed(
                batch_size      = len(batch),
                current_model   = self.queue.currently_loaded_model,
                affinity_hits   = sum(
                    1 for r in batch
                    if self.queue.currently_loaded_model and
                    normalise_model_name(r.model) ==
                    normalise_model_name(self.queue.currently_loaded_model)
                ),
                models_in_batch = [r.model for r in batch],
                remaining_queue = self.queue.size,
            )

            logger.info(
                "[SCHEDULER] Dispatching batch  size=%d  models=%s",
                len(batch), list({r.model for r in batch}),
            )

            for req in batch:
                if not self._running:
                    req.status = RequestStatus.CANCELLED
                    await req.stream_queue.put(b'{"error":"Scheduler shutting down"}\n')
                    await req.stream_queue.put(None)
                    continue
                await self._execute_one(req)

    async def _execute_one(self, req: PendingRequest) -> None:
        current  = self.queue.currently_loaded_model
        switching = (
            current is not None and
            normalise_model_name(current) != normalise_model_name(req.model)
        )
        if switching:
            await self._handle_model_switch(current, req.model)  # type: ignore

        if not await self.ollama.check_model_available(req.model):
            msg = json.dumps({"error": f"Model '{req.model}' not found. Run: ollama pull {req.model}"}).encode() + b"\n"
            req.status = RequestStatus.FAILED
            await req.stream_queue.put(msg)
            await req.stream_queue.put(None)
            return

        payload = {**req.payload, "keep_alive": settings.KEEP_ALIVE_ACTIVE, "stream": True}

        # ── Pure execution latency starts here ────────────────────────────────
        req.execution_start_time = time.monotonic()
        req.status = RequestStatus.EXECUTING
        self.queue.currently_loaded_model = req.model

        logger.info(
            "[EXEC START] req=%-8s  model=%-22s  queue_wait=%.1f ms",
            req.request_id[:8], req.model, req.queue_wait_ms or 0,
        )

        try:
            async for raw_chunk in self.ollama.stream_generate(req.endpoint, payload):
                await req.stream_queue.put(raw_chunk)
                try:
                    data = json.loads(raw_chunk)
                    if data.get("eval_count"):
                        req.tokens_generated = int(data["eval_count"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
            req.status = RequestStatus.COMPLETED
        except Exception as exc:
            logger.error("[EXEC ERROR] req=%s  error=%s", req.request_id[:8], exc)
            await req.stream_queue.put(json.dumps({"error": str(exc)}).encode() + b"\n")
            req.status = RequestStatus.FAILED
            req.error  = exc
        finally:
            req.execution_end_time = time.monotonic()
            await req.stream_queue.put(None)
            latency_tracker.record(req)
            self.queue.total_processed += 1

    async def _handle_model_switch(self, current_model: str, new_model: str) -> None:
        cur_vram = extract_vram_gb(current_model)
        new_vram = extract_vram_gb(new_model)
        combined = cur_vram + new_vram
        force    = combined > settings.VRAM_BUDGET_GB

        logger.info(
            "[MODEL SWITCH] %s (%.1f GB) → %s (%.1f GB)  combined=%.1f GB  budget=%.1f GB",
            current_model, cur_vram, new_model, new_vram, combined, settings.VRAM_BUDGET_GB,
        )

        # ── Log switch start ──────────────────────────────────────────────────
        analytics.log_switch_start(
            from_model   = current_model,
            to_model     = new_model,
            from_vram_gb = cur_vram,
            to_vram_gb   = new_vram,
            force_unload = force,
        )

        if force:
            logger.warning(
                "⚠️  Combined VRAM (%.1f GB) exceeds budget. Force-unloading %s.",
                combined, current_model,
            )
            await self.ollama.unload_model(current_model)
            await asyncio.sleep(1.0)

        # ── Log switch end ────────────────────────────────────────────────────
        analytics.log_switch_end(to_model=new_model)