"""
scheduler.py — ModelScheduler worker and LatencyTracker.

ModelScheduler
──────────────
Runs as a single long-lived asyncio Task.  It is intentionally *single-threaded*
(one request executes at a time) so that model switches are atomic and VRAM
state is always deterministic.

Execution lifecycle for each request
─────────────────────────────────────
  1.  Wait for queue to be non-empty.
  2.  Sleep BATCH_COLLECT_TIMEOUT_S to let additional same-model requests
      arrive and be coalesced into the same batch window.
  3.  Pull a reordered batch from RequestQueue.get_batch().
  4.  For each PendingRequest in the batch:
      a.  Check if the model has changed since the last request.
      b.  If changing to a model that would exceed VRAM_BUDGET_GB in combination
          with the currently-loaded model: call OllamaClient.unload_model().
      c.  Verify the model exists locally (ollama list check).
      d.  Stamp execution_start_time  ← pure execution latency begins here.
      e.  Stream Ollama response; pipe each chunk to request.stream_queue.
      f.  Stamp execution_end_time.
      g.  Place None sentinel on stream_queue to signal end-of-stream.
      h.  Call LatencyTracker.record() and log structured metrics.

LatencyTracker
──────────────
In-memory ring-buffer (default 1 000 records).  Provides per-model P50/P95/P99
execution latency and tokens-per-second aggregates for the /metrics endpoint.
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

logger = logging.getLogger(__name__)


# ── LatencyTracker ────────────────────────────────────────────────────────────

class LatencyTracker:
    """
    Append-only ring buffer of LatencyRecord objects.

    Thread-safety: all writes happen inside the single scheduler Task, so no
    locking is required.  Reads from FastAPI route handlers are safe because
    Python's GIL protects list operations of this size.
    """

    def __init__(self, max_records: int = 1_000):
        self._records: list[LatencyRecord] = []
        self._max = max_records

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, req: PendingRequest) -> None:
        """Snapshot a completed PendingRequest into a LatencyRecord."""
        if req.execution_latency_ms is None:
            return   # request never executed (was cancelled / dropped)

        exec_ms  = req.execution_latency_ms
        tok_s    = (req.tokens_generated / (exec_ms / 1000.0)
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

        # Structured log line — easy to ship to CloudWatch / Datadog
        logger.info(
            "[LATENCY] req=%-8s model=%-20s "
            "queue_wait=%7.1f ms  exec=%8.1f ms  total=%8.1f ms  "
            "tokens=%4d  tok/s=%6.1f  status=%s",
            rec.request_id[:8],
            rec.model,
            rec.queue_wait_ms,
            rec.execution_ms,
            rec.total_ms,
            rec.tokens_generated,
            rec.tokens_per_second,
            rec.status,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_summary(self, model: Optional[str] = None) -> dict:
        """
        Return aggregate statistics for all records, or filtered to one model.
        Percentiles use the "nearest rank" method (good enough for dashboards).
        """
        recs = [r for r in self._records if model is None or r.model == model]
        if not recs:
            return {"count": 0}

        def pct(lst: list[float], p: float) -> float:
            idx = max(0, int(len(lst) * p / 100) - 1)
            return round(sorted(lst)[idx], 1)

        exec_ms   = [r.execution_ms      for r in recs]
        wait_ms   = [r.queue_wait_ms     for r in recs]
        tok_s_lst = [r.tokens_per_second for r in recs]

        return {
            "count":              len(recs),
            "avg_exec_ms":        round(sum(exec_ms)   / len(exec_ms),   1),
            "p50_exec_ms":        pct(exec_ms,  50),
            "p95_exec_ms":        pct(exec_ms,  95),
            "p99_exec_ms":        pct(exec_ms,  99),
            "avg_queue_wait_ms":  round(sum(wait_ms)   / len(wait_ms),   1),
            "avg_tok_per_s":      round(sum(tok_s_lst) / len(tok_s_lst), 1),
        }

    def all_models(self) -> set[str]:
        return {r.model for r in self._records}


# Module-level singleton accessed by main.py
latency_tracker = LatencyTracker()


# ── ModelScheduler ────────────────────────────────────────────────────────────

class ModelScheduler:
    """
    Single-worker asyncio scheduler.

    Design invariant: only ONE Ollama request is in-flight at any moment.
    This keeps VRAM state deterministic and makes model-switch logic trivial.
    """

    def __init__(self, queue: RequestQueue, ollama: OllamaClient):
        self.queue  = queue
        self.ollama = ollama
        self._running     = False
        self._worker_task: Optional[asyncio.Task] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running     = True
        self._worker_task = asyncio.create_task(
            self._run_loop(), name="scheduler-worker"
        )
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
        logger.info("🛑  ModelScheduler stopped cleanly")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        logger.info("[SCHEDULER] Event loop running")

        while self._running:
            # ── Wait for work ─────────────────────────────────────────────────
            try:
                await asyncio.wait_for(
                    self.queue.wait_for_items(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue   # no work yet; loop back

            # ── Collection window ─────────────────────────────────────────────
            # Sleep briefly so that a burst of same-model requests that arrive
            # within BATCH_COLLECT_TIMEOUT_S are all visible to get_batch(),
            # improving affinity grouping without meaningfully harming latency.
            await asyncio.sleep(settings.BATCH_COLLECT_TIMEOUT_S)

            # ── Pull reordered batch ──────────────────────────────────────────
            batch = await self.queue.get_batch()
            if not batch:
                continue

            logger.info(
                "[SCHEDULER] Dispatching batch  size=%d  models=%s",
                len(batch),
                list({r.model for r in batch}),
            )

            # ── Execute each request serially ─────────────────────────────────
            for req in batch:
                if not self._running:
                    # Scheduler is shutting down; drain batch with cancellations
                    req.status = RequestStatus.CANCELLED
                    await req.stream_queue.put(
                        b'{"error":"Scheduler is shutting down"}\n'
                    )
                    await req.stream_queue.put(None)
                    continue

                await self._execute_one(req)

    # ── Per-request execution ─────────────────────────────────────────────────

    async def _execute_one(self, req: PendingRequest) -> None:
        """
        Execute a single PendingRequest against Ollama.

        Steps
        ─────
        1.  Model switch detection + VRAM management.
        2.  Model availability check.
        3.  Payload finalisation (inject keep_alive, force stream=True).
        4.  Stamp execution_start_time  ← pure latency clock starts.
        5.  Stream response, pipe bytes to req.stream_queue.
        6.  Stamp execution_end_time  ← pure latency clock stops.
        7.  Signal end-of-stream with None sentinel.
        8.  Record latency.
        """

        # ── 1. Model switch ───────────────────────────────────────────────────
        current = self.queue.currently_loaded_model
        switching = (
            current is not None
            and normalise_model_name(current) != normalise_model_name(req.model)
        )
        if switching:
            await self._handle_model_switch(current, req.model)  # type: ignore[arg-type]

        # ── 2. Availability check ─────────────────────────────────────────────
        if not await self.ollama.check_model_available(req.model):
            msg = json.dumps(
                {"error": f"Model '{req.model}' not found. Run: ollama pull {req.model}"}
            ).encode() + b"\n"
            req.status = RequestStatus.FAILED
            req.error  = ValueError(f"Model not available: {req.model}")
            await req.stream_queue.put(msg)
            await req.stream_queue.put(None)
            logger.error("[SCHEDULER] Model not found: %s", req.model)
            return

        # ── 3. Finalise payload ───────────────────────────────────────────────
        payload = {
            **req.payload,
            "keep_alive": settings.KEEP_ALIVE_ACTIVE,   # keep warm while queue active
            "stream":     True,                          # always stream internally
        }

        # ── 4. Execution start ────────────────────────────────────────────────
        req.execution_start_time = time.monotonic()
        req.status = RequestStatus.EXECUTING
        # Update queue's view of the world so the next get_batch() sorts correctly
        self.queue.currently_loaded_model = req.model

        logger.info(
            "[EXEC START] req=%-8s  model=%-20s  queue_wait=%.1f ms",
            req.request_id[:8],
            req.model,
            req.queue_wait_ms or 0,
        )

        # ── 5. Stream from Ollama ─────────────────────────────────────────────
        try:
            async for raw_chunk in self.ollama.stream_generate(req.endpoint, payload):
                await req.stream_queue.put(raw_chunk)

                # Best-effort token count extraction for metrics
                try:
                    data = json.loads(raw_chunk)
                    if data.get("eval_count"):
                        req.tokens_generated = int(data["eval_count"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

            req.status = RequestStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "[EXEC ERROR] req=%s  error=%s", req.request_id[:8], exc
            )
            err_bytes = json.dumps({"error": str(exc)}).encode() + b"\n"
            await req.stream_queue.put(err_bytes)
            req.status = RequestStatus.FAILED
            req.error  = exc

        finally:
            # ── 6. Execution end ──────────────────────────────────────────────
            req.execution_end_time = time.monotonic()

            # ── 7. End-of-stream sentinel ─────────────────────────────────────
            await req.stream_queue.put(None)

            # ── 8. Latency accounting ─────────────────────────────────────────
            latency_tracker.record(req)
            self.queue.total_processed += 1

    # ── VRAM management ───────────────────────────────────────────────────────

    async def _handle_model_switch(
        self, current_model: str, new_model: str
    ) -> None:
        """
        Decide whether to explicitly unload `current_model` before loading
        `new_model`.

        Decision rule
        ─────────────
        If  vram(current) + vram(new)  > VRAM_BUDGET_GB:
            Force-unload current via keep_alive=0 and wait 1 s for the
            CUDA allocator to reclaim pages before Ollama starts loading
            the new model.
        Else:
            Trust Ollama's own LRU eviction — it will free enough space.

        Why 1 s sleep after unload?
        Ollama signals "unloaded" when it has released its Go-side references,
        but the CUDA driver may not have fully reclaimed pages yet.  A 1 s
        grace period prevents transient OOM errors on the A10G when the
        combined size is close to the 22 GB budget.
        """
        cur_vram = extract_vram_gb(current_model)
        new_vram = extract_vram_gb(new_model)
        combined = cur_vram + new_vram

        logger.info(
            "[MODEL SWITCH] %s (%.1f GB) → %s (%.1f GB)  "
            "combined=%.1f GB  budget=%.1f GB",
            current_model, cur_vram,
            new_model,     new_vram,
            combined,      settings.VRAM_BUDGET_GB,
        )

        if combined > settings.VRAM_BUDGET_GB:
            logger.warning(
                "⚠️  Combined VRAM (%.1f GB) exceeds budget (%.1f GB). "
                "Force-unloading %s before loading %s.",
                combined, settings.VRAM_BUDGET_GB,
                current_model, new_model,
            )
            await self.ollama.unload_model(current_model)
            await asyncio.sleep(1.0)   # grace period for CUDA reclaim
        else:
            logger.debug(
                "VRAM headroom OK (%.1f GB remaining). "
                "Letting Ollama manage eviction.",
                settings.VRAM_BUDGET_GB - cur_vram,
            )
