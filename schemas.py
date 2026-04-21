"""
schemas.py — Data-transfer objects and internal request envelope.

PendingRequest is the single unit of work that travels through the system:
    Client → FastAPI → RequestQueue → ModelScheduler → OllamaClient → Client

Timing model
────────────
  wall_enqueue_time  ──► execution_start_time ──► execution_end_time
  |←── queue_wait ──────►|←──── pure_execution_latency ────────────►|
  |←──────────────── total_latency ──────────────────────────────────►|

"Pure execution latency" is started only when the worker begins streaming
from Ollama — queue wait time is intentionally excluded so that metrics
reflect true model performance rather than scheduler contention.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RequestStatus(str, Enum):
    QUEUED    = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


@dataclass
class PendingRequest:
    """Internal envelope wrapping one /api/generate or /api/chat call."""

    model:    str                          # e.g. "llama3.1:8b"
    endpoint: str                          # "/api/generate" | "/api/chat"
    payload:  dict                         # original decoded JSON body
    streaming: bool = True

    # Auto-assigned identity & timing
    request_id:         str   = field(default_factory=lambda: str(uuid.uuid4()))
    enqueue_time:       float = field(default_factory=time.monotonic)
    wall_enqueue_time:  float = field(default_factory=time.time)
    status: RequestStatus     = RequestStatus.QUEUED

    # Set by the worker when it begins / finishes the Ollama call
    execution_start_time: Optional[float] = None
    execution_end_time:   Optional[float] = None

    # Per-request asyncio.Queue used to pipe streaming chunks to the client.
    # The worker puts bytes chunks here; None is the end-of-stream sentinel.
    stream_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    error:            Optional[Exception] = None
    tokens_generated: int = 0              # populated from Ollama's eval_count

    # ── Derived latency helpers ───────────────────────────────────────────────

    @property
    def queue_wait_ms(self) -> Optional[float]:
        """Milliseconds spent waiting in the queue before execution began."""
        if self.execution_start_time is not None:
            return (self.execution_start_time - self.enqueue_time) * 1000
        return None

    @property
    def execution_latency_ms(self) -> Optional[float]:
        """
        PURE execution latency: time the model was actively processing this
        request, excluding any queue wait.  This is the primary performance KPI.
        """
        if self.execution_start_time is not None and self.execution_end_time is not None:
            return (self.execution_end_time - self.execution_start_time) * 1000
        return None

    @property
    def total_latency_ms(self) -> Optional[float]:
        """End-to-end latency from enqueue to last byte delivered."""
        if self.execution_end_time is not None:
            return (self.execution_end_time - self.enqueue_time) * 1000
        return None


@dataclass
class LatencyRecord:
    """Immutable snapshot of a completed request, stored in LatencyTracker."""

    request_id:        str
    model:             str
    endpoint:          str
    enqueue_wall_time: float    # Unix timestamp of enqueue
    queue_wait_ms:     float
    execution_ms:      float    # pure execution latency
    total_ms:          float
    tokens_generated:  int   = 0
    tokens_per_second: float = 0.0
    status:            str   = "completed"
