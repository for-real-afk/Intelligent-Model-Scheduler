"""
queue_manager.py — Model-affinity batch queue.

Core idea (anti-thrash scheduling)
────────────────────────────────────
When the A10G has a 26B model loaded and new requests trickle in for both
2B and 26B models, naively serving them FIFO causes:
    load 26B → load 2B → load 26B → load 2B  (GPU thrashing)

The RequestQueue prevents this by reordering each batch window:

    Incoming FIFO:   [2B, 26B, 2B, 26B, 26B, 2B]
    Currently loaded: 26B
    After reorder:   [26B, 26B, 26B, 2B, 2B, 2B]
                      ↑ affinity group   ↑ deferred group

Result: the loaded model's requests drain completely before the GPU pays
the cost of a model swap.  With a 10-item window on a busy instance, a
single 26B → 2B swap becomes at most one swap per 10 requests instead of
one swap per request.

Public API
──────────
    await queue.enqueue(req)    → bool (False = queue full / back-pressure)
    await queue.wait_for_items() → blocks until at least one item is present
    await queue.get_batch()     → List[PendingRequest], reordered by affinity
    queue.get_queue_snapshot()  → read-only view for /queue/status endpoint
"""
import asyncio
import logging
import re
import time
from collections import deque
from typing import Optional

from config import settings, MODEL_VRAM_MAP
from schemas import PendingRequest

logger = logging.getLogger(__name__)


# ── Utility helpers ───────────────────────────────────────────────────────────

def normalise_model_name(name: str) -> str:
    """Lowercase + strip whitespace.  Used for equality comparisons."""
    return name.strip().lower()


def extract_vram_gb(model_name: str) -> float:
    """
    Heuristically estimate VRAM requirement (GB, 4-bit quant) from a model name.

    Examples
    ────────
    "llama3.1:8b"          → 5.5 GB
    "qwen2.5-coder:26b-q4" → 17.0 GB
    "mistral:7b-instruct"  → 4.5 GB
    "unknown-model"        → 4.0 GB  (safe fallback)
    """
    name_lower = model_name.lower()
    match = re.search(r"(\d+(?:\.\d+)?)b", name_lower)
    if not match:
        return 4.0  # conservative default

    param_billions = float(match.group(1))
    # Find closest entry in our VRAM table
    closest_key = min(
        MODEL_VRAM_MAP,
        key=lambda k: abs(float(k.replace("b", "")) - param_billions),
    )
    return MODEL_VRAM_MAP[closest_key]


# ── RequestQueue ──────────────────────────────────────────────────────────────

class RequestQueue:
    """
    Asyncio-safe FIFO queue with batch-window model-affinity reordering.

    Internal storage is a plain collections.deque (O(1) append/popleft).
    An asyncio.Event provides zero-cost blocking when the queue is empty.
    A single asyncio.Lock serialises all structural mutations.
    """

    def __init__(self):
        self._deque: deque[PendingRequest] = deque()
        self._lock       = asyncio.Lock()
        self._not_empty  = asyncio.Event()

        # Maintained by the scheduler worker after each model switch
        self._currently_loaded_model: Optional[str] = None

        # Counters for the /health endpoint
        self.total_enqueued:  int = 0
        self.total_processed: int = 0
        self.total_dropped:   int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._deque)

    @property
    def currently_loaded_model(self) -> Optional[str]:
        return self._currently_loaded_model

    @currently_loaded_model.setter
    def currently_loaded_model(self, model: Optional[str]):
        if model != self._currently_loaded_model:
            logger.info(
                f"[QUEUE] Loaded model updated: "
                f"{self._currently_loaded_model!r} → {model!r}"
            )
        self._currently_loaded_model = model

    # ── Enqueue ───────────────────────────────────────────────────────────────

    async def enqueue(self, request: PendingRequest) -> bool:
        """
        Add a request to the tail of the queue.

        Returns False (back-pressure signal) when the queue is at capacity so
        that the FastAPI layer can return HTTP 503 instead of silently dropping.
        """
        async with self._lock:
            if len(self._deque) >= settings.MAX_QUEUE_SIZE:
                logger.warning(
                    f"[QUEUE FULL] Dropping req={request.request_id[:8]} "
                    f"model={request.model} "
                    f"depth={len(self._deque)}/{settings.MAX_QUEUE_SIZE}"
                )
                self.total_dropped += 1
                return False

            self._deque.append(request)
            self.total_enqueued += 1
            logger.debug(
                f"[ENQUEUE] req={request.request_id[:8]} model={request.model} "
                f"depth={len(self._deque)}"
            )

        # Signal waiting worker(s) that work is available
        self._not_empty.set()
        return True

    # ── Dequeue (batch) ───────────────────────────────────────────────────────

    async def wait_for_items(self):
        """Suspend the caller until at least one item is present in the queue."""
        await self._not_empty.wait()

    async def get_batch(self) -> list[PendingRequest]:
        """
        Pull up to BATCH_WINDOW_SIZE items and re-order by model affinity.

        Algorithm
        ─────────
        1.  Take min(BATCH_WINDOW_SIZE, queue_depth) items from the front.
        2.  Partition into two stable lists:
                • affinity  — requests whose model == currently_loaded_model
                • deferred  — all other requests
        3.  Return affinity + deferred.

        The stable partition preserves FIFO ordering within each group, which
        is important for fairness: the oldest 26B request is always served
        before the newest 26B request.
        """
        async with self._lock:
            n = min(settings.BATCH_WINDOW_SIZE, len(self._deque))
            if n == 0:
                return []

            window: list[PendingRequest] = [self._deque.popleft() for _ in range(n)]

            if not self._deque:
                self._not_empty.clear()   # queue is now empty; block next wait

            current = self._currently_loaded_model
            if current:
                current_norm = normalise_model_name(current)
                affinity  = [r for r in window if normalise_model_name(r.model) == current_norm]
                deferred  = [r for r in window if normalise_model_name(r.model) != current_norm]
                batch = affinity + deferred
            else:
                batch = window

            affinity_count = len(batch) - len(
                [r for r in batch if normalise_model_name(r.model) != (
                    normalise_model_name(current) if current else ""
                )]
            )

            logger.info(
                f"[BATCH FORMED] size={len(batch)}  "
                f"current_model={current!r}  "
                f"affinity_hits={affinity_count}  "
                f"remaining_queue={len(self._deque)}"
            )

            return batch

    # ── Observability ─────────────────────────────────────────────────────────

    def get_queue_snapshot(self) -> list[dict]:
        """
        Non-destructive read of the current queue for the /queue/status endpoint.
        No lock is acquired (eventual consistency is fine for monitoring).
        """
        now = time.monotonic()
        return [
            {
                "request_id":   r.request_id[:8],
                "model":        r.model,
                "status":       r.status.value,
                "waiting_ms":   round((now - r.enqueue_time) * 1000, 1),
            }
            for r in self._deque
        ]


# Module-level singleton
request_queue = RequestQueue()
