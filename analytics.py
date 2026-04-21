"""
analytics.py — Structured JSONL logger for comparative analysis.

Every significant event is written as a single JSON line to:
    ./logs/scheduler_events.jsonl

Event types
───────────
  request_complete   — full latency breakdown per request
  batch_formed       — batch composition and affinity stats
  model_switch       — VRAM cost and switch duration
  model_cold_load    — first-request EBS/NVMe load time per model
  scheduler_start    — startup snapshot
  hourly_summary     — rolling per-model aggregates (written every hour)

Analysis
────────
  Run:  python analytics.py --report
  to get a printed comparative table across all models logged so far.

  Run:  python analytics.py --export csv
  to write logs/summary.csv for Excel / pandas / Google Sheets.
"""

import asyncio
import csv
import json
import logging
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "scheduler_events.jsonl"
LOG_DIR.mkdir(exist_ok=True)


# ── Writer ────────────────────────────────────────────────────────────────────

class AnalyticsLogger:
    """
    Appends structured JSON events to a JSONL file.
    Also maintains an in-memory index for the /analytics endpoint.
    """

    def __init__(self):
        self._fh = open(LOG_FILE, "a", buffering=1)   # line-buffered
        self._records: list[dict] = []
        self._model_first_seen: dict[str, float] = {}  # model → first exec start
        self._switch_start: Optional[float] = None
        self._switch_from:  Optional[str]   = None
        logger.info(f"📊  Analytics logger writing to {LOG_FILE.resolve()}")

    def _write(self, event: dict):
        event["ts"] = datetime.now(timezone.utc).isoformat()
        self._fh.write(json.dumps(event) + "\n")
        self._records.append(event)

    # ── Public event methods ──────────────────────────────────────────────────

    def log_startup(self, config: dict):
        self._write({"event": "scheduler_start", **config})

    def log_batch_formed(
        self,
        batch_size:     int,
        current_model:  Optional[str],
        affinity_hits:  int,
        models_in_batch: list[str],
        remaining_queue: int,
    ):
        self._write({
            "event":           "batch_formed",
            "batch_size":      batch_size,
            "current_model":   current_model,
            "affinity_hits":   affinity_hits,
            "affinity_pct":    round(affinity_hits / batch_size * 100, 1) if batch_size else 0,
            "models_in_batch": list(set(models_in_batch)),
            "unique_models":   len(set(models_in_batch)),
            "remaining_queue": remaining_queue,
        })

    def log_switch_start(self, from_model: str, to_model: str,
                         from_vram_gb: float, to_vram_gb: float,
                         force_unload: bool):
        self._switch_start = time.monotonic()
        self._switch_from  = from_model
        self._write({
            "event":        "model_switch_start",
            "from_model":   from_model,
            "to_model":     to_model,
            "from_vram_gb": from_vram_gb,
            "to_vram_gb":   to_vram_gb,
            "combined_gb":  round(from_vram_gb + to_vram_gb, 1),
            "force_unload": force_unload,
        })

    def log_switch_end(self, to_model: str):
        duration_ms = (
            round((time.monotonic() - self._switch_start) * 1000, 1)
            if self._switch_start else None
        )
        self._write({
            "event":          "model_switch_end",
            "from_model":     self._switch_from,
            "to_model":       to_model,
            "switch_duration_ms": duration_ms,
        })
        self._switch_start = None
        self._switch_from  = None

    def log_request_complete(
        self,
        request_id:     str,
        model:          str,
        endpoint:       str,
        queue_wait_ms:  float,
        exec_ms:        float,
        total_ms:       float,
        tokens:         int,
        tok_per_s:      float,
        status:         str,
        streaming:      bool,
        was_cold_load:  bool,
    ):
        self._write({
            "event":          "request_complete",
            "request_id":     request_id[:8],
            "model":          model,
            "endpoint":       endpoint,
            "queue_wait_ms":  round(queue_wait_ms, 1),
            "exec_ms":        round(exec_ms, 1),
            "total_ms":       round(total_ms, 1),
            "tokens":         tokens,
            "tok_per_s":      round(tok_per_s, 2),
            "status":         status,
            "streaming":      streaming,
            "was_cold_load":  was_cold_load,
        })

    def mark_model_seen(self, model: str, exec_start: float) -> bool:
        """
        Returns True (cold load) the first time a model is executed.
        Returns False on all subsequent calls.
        """
        if model not in self._model_first_seen:
            self._model_first_seen[model] = exec_start
            return True
        return False

    # ── Hourly summary task ───────────────────────────────────────────────────

    async def run_hourly_summary(self):
        """Write a rolling hourly summary event. Run as a background task."""
        while True:
            await asyncio.sleep(3600)
            summary = self._build_summary()
            self._write({"event": "hourly_summary", **summary})
            logger.info(f"📊  Hourly summary written to {LOG_FILE}")

    # ── In-memory analysis ────────────────────────────────────────────────────

    def _build_summary(self) -> dict:
        requests = [r for r in self._records if r["event"] == "request_complete"]
        switches = [r for r in self._records if r["event"] == "model_switch_end"]

        per_model: dict[str, list] = defaultdict(list)
        for r in requests:
            per_model[r["model"]].append(r)

        model_stats = {}
        for model, reqs in per_model.items():
            warm  = [r for r in reqs if not r.get("was_cold_load")]
            cold  = [r for r in reqs if r.get("was_cold_load")]
            exec_times = [r["exec_ms"] for r in warm] or [0]
            tok_rates  = [r["tok_per_s"] for r in warm if r["tok_per_s"] > 0] or [0]

            model_stats[model] = {
                "total_requests":     len(reqs),
                "warm_requests":      len(warm),
                "cold_loads":         len(cold),
                "avg_exec_ms":        round(statistics.mean(exec_times), 1),
                "p50_exec_ms":        round(statistics.median(exec_times), 1),
                "p95_exec_ms":        round(sorted(exec_times)[int(len(exec_times)*.95)], 1),
                "avg_tok_per_s":      round(statistics.mean(tok_rates), 2),
                "total_tokens":       sum(r["tokens"] for r in reqs),
                "avg_cold_load_ms":   round(
                    statistics.mean([r["exec_ms"] for r in cold]), 1
                ) if cold else None,
            }

        return {
            "total_requests":  len(requests),
            "total_switches":  len(switches),
            "per_model":       model_stats,
        }

    def get_summary(self) -> dict:
        return self._build_summary()

    def close(self):
        self._fh.close()


# Module-level singleton
analytics = AnalyticsLogger()


# ── CLI report ────────────────────────────────────────────────────────────────

def _print_report():
    if not LOG_FILE.exists():
        print("No log file found. Run the scheduler and make some requests first.")
        return

    records = [json.loads(l) for l in LOG_FILE.read_text().splitlines() if l.strip()]
    requests = [r for r in records if r["event"] == "request_complete"]
    switches = [r for r in records if r["event"] == "model_switch_end"]
    batches  = [r for r in records if r["event"] == "batch_formed"]

    if not requests:
        print("No completed requests in log yet.")
        return

    per_model: dict[str, list] = defaultdict(list)
    for r in requests:
        per_model[r["model"]].append(r)

    print("\n" + "═"*90)
    print("  OLLAMA SCHEDULER — COMPARATIVE ANALYSIS REPORT")
    print(f"  Log file: {LOG_FILE}  |  Total events: {len(records)}")
    print("═"*90)

    # ── Per-model table ───────────────────────────────────────────────────────
    header = f"{'Model':<25} {'Reqs':>5} {'Cold':>5} {'Warm':>5} {'Avg Exec':>10} {'P50':>8} {'P95':>8} {'tok/s':>7} {'Tokens':>8}"
    print(f"\n{'─'*90}")
    print(header)
    print(f"{'─'*90}")

    for model, reqs in sorted(per_model.items()):
        warm  = [r for r in reqs if not r.get("was_cold_load")]
        cold  = [r for r in reqs if r.get("was_cold_load")]
        exec_t = sorted([r["exec_ms"] for r in warm]) or [0]
        toks   = [r["tok_per_s"] for r in warm if r["tok_per_s"] > 0] or [0]
        p95_i  = max(0, int(len(exec_t) * 0.95) - 1)

        print(
            f"{model:<25} "
            f"{len(reqs):>5} "
            f"{len(cold):>5} "
            f"{len(warm):>5} "
            f"{statistics.mean(exec_t):>9.0f}ms "
            f"{statistics.median(exec_t):>7.0f}ms "
            f"{exec_t[p95_i]:>7.0f}ms "
            f"{statistics.mean(toks):>6.1f} "
            f"{sum(r['tokens'] for r in reqs):>8}"
        )

    # ── Cold load comparison ──────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("  COLD LOAD TIMES (EBS/NVMe → VRAM)")
    print(f"{'─'*90}")
    for model, reqs in sorted(per_model.items()):
        cold = [r for r in reqs if r.get("was_cold_load")]
        if cold:
            for c in cold:
                print(f"  {model:<25}  {c['exec_ms']:>8.0f}ms  ({c['ts'][:19]})")

    # ── Model switch analysis ─────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("  MODEL SWITCH ANALYSIS")
    print(f"{'─'*90}")
    print(f"  Total switches:       {len(switches)}")
    if switches:
        durations = [s["switch_duration_ms"] for s in switches if s.get("switch_duration_ms")]
        if durations:
            print(f"  Avg switch duration:  {statistics.mean(durations):.0f}ms")
        forced = sum(1 for s in records
                     if s["event"] == "model_switch_start" and s.get("force_unload"))
        print(f"  Force-unloads fired:  {forced}")
        print(f"\n  {'From':<25} {'To':<25} {'Duration':>10} {'Force':>7}")
        print(f"  {'─'*70}")
        for s in switches[-10:]:   # last 10 switches
            dur = f"{s['switch_duration_ms']:.0f}ms" if s.get("switch_duration_ms") else "n/a"
            # find matching start event for force_unload flag
            print(f"  {(s.get('from_model') or 'None'):<25} {s['to_model']:<25} {dur:>10}")

    # ── Batch affinity ────────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("  BATCH AFFINITY EFFICIENCY")
    print(f"{'─'*90}")
    if batches:
        all_affinity_pcts = [b["affinity_pct"] for b in batches]
        multi_model = [b for b in batches if b["unique_models"] > 1]
        print(f"  Total batches formed:      {len(batches)}")
        print(f"  Avg affinity hit rate:     {statistics.mean(all_affinity_pcts):.1f}%")
        print(f"  Mixed-model batches:       {len(multi_model)}")
        print(f"  Avg batch size:            {statistics.mean(b['batch_size'] for b in batches):.1f}")

    print(f"\n{'═'*90}\n")


def _export_csv():
    if not LOG_FILE.exists():
        print("No log file found.")
        return

    records = [json.loads(l) for l in LOG_FILE.read_text().splitlines() if l.strip()]
    requests = [r for r in records if r["event"] == "request_complete"]

    out = LOG_DIR / "summary.csv"
    with open(out, "w", newline="") as f:
        fields = ["ts","request_id","model","endpoint","queue_wait_ms",
                  "exec_ms","total_ms","tokens","tok_per_s","status",
                  "streaming","was_cold_load"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(requests)

    print(f"✅  Exported {len(requests)} request records to {out.resolve()}")


if __name__ == "__main__":
    import sys
    if "--export" in sys.argv and "csv" in sys.argv:
        _export_csv()
    else:
        _print_report()
