"""
main.py — FastAPI application entry point.

Traffic path
────────────
  Nginx (port 80)
      ↓  proxy_pass
  FastAPI / Uvicorn (port 8080)   ← this file
      ↓  enqueue PendingRequest
  RequestQueue  (queue_manager.py)
      ↓  batch-window pop
  ModelScheduler worker  (scheduler.py)
      ↓  httpx streaming
  Ollama  (port 11434, localhost)

Routes
──────
  POST /api/generate      → queued, streamed back to client
  POST /api/chat          → queued, streamed back to client
  GET  /health            → liveness probe (used by ALB / ECS)
  GET  /metrics           → latency stats, per-model aggregates
  GET  /queue/status      → real-time queue depth + pending request list
  GET  /models            → available + running models from Ollama
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from ollama_client import ollama
from queue_manager import request_queue
from scheduler import ModelScheduler, latency_tracker
from schemas import PendingRequest, RequestStatus

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)-24s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Scheduler singleton ────────────────────────────────────────────────────────
scheduler = ModelScheduler(queue=request_queue, ollama=ollama)


# ── App lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on boot; stop it on shutdown."""
    logger.info(
        "Starting Ollama Model Scheduler  "
        "ollama=%s  batch_size=%d  vram_budget=%.0fGB",
        settings.OLLAMA_BASE_URL,
        settings.BATCH_WINDOW_SIZE,
        settings.VRAM_BUDGET_GB,
    )
    await scheduler.start()
    yield
    logger.info("Shutting down scheduler …")
    await scheduler.stop()


app = FastAPI(
    title="Ollama Intelligent Model Scheduler",
    description=(
        "Batch-windowed, model-affinity scheduler for Ollama on AWS g5.2xlarge. "
        "Minimises GPU VRAM thrashing via affinity-grouped batch windows."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Core proxy helper ──────────────────────────────────────────────────────────

async def _enqueue_and_stream(
    endpoint: str,
    raw_body: dict,
) -> StreamingResponse:
    """
    Create a PendingRequest, enqueue it, and return a StreamingResponse that
    yields NDJSON chunks as the scheduler worker processes the request.

    Client-visible headers
    ──────────────────────
    X-Request-ID      : UUID of this request (for tracing in logs)
    X-Queue-Depth     : queue depth at the moment this request was accepted
    """
    model = raw_body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="'model' field is required in request body.")

    req = PendingRequest(
        model    = model,
        endpoint = endpoint,
        payload  = raw_body,
        streaming= raw_body.get("stream", True),
    )

    accepted = await request_queue.enqueue(req)
    if not accepted:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Scheduler queue is at capacity ({settings.MAX_QUEUE_SIZE} requests). "
                "Retry after a short delay."
            ),
        )

    queue_depth_at_accept = request_queue.size

    # ── Streaming generator ────────────────────────────────────────────────────
    async def _generate() -> AsyncGenerator[bytes, None]:
        """
        Yield bytes from req.stream_queue until the None sentinel is received,
        or until REQUEST_TIMEOUT_S elapses (whichever comes first).

        Using asyncio.wait_for with a timeout on each individual .get() call
        allows the generator to detect abandoned connections promptly without
        blocking the event loop.
        """
        deadline = time.monotonic() + settings.REQUEST_TIMEOUT_S

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "[TIMEOUT] req=%s model=%s timed out after %.0f s",
                    req.request_id[:8], model, settings.REQUEST_TIMEOUT_S,
                )
                yield b'{"error":"Request timed out in scheduler queue or execution"}\n'
                return

            try:
                # Use a 30 s per-get timeout; if nothing arrives we loop back
                # and recheck the global deadline.
                chunk = await asyncio.wait_for(
                    req.stream_queue.get(),
                    timeout=min(remaining, 30.0),
                )
            except asyncio.TimeoutError:
                # Still waiting for the worker — keep looping
                continue

            if chunk is None:
                # End-of-stream sentinel placed by the scheduler worker
                return

            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="application/x-ndjson",
        headers={
            "X-Request-ID":  req.request_id,
            "X-Queue-Depth": str(queue_depth_at_accept),
        },
    )


# ── Inference routes ───────────────────────────────────────────────────────────

@app.post(
    "/api/generate",
    summary="Text generation (queued + model-affinity scheduled)",
    response_description="NDJSON stream from Ollama /api/generate",
)
async def generate(request: Request):
    """
    Drop-in replacement for `POST http://localhost:11434/api/generate`.
    Accepts the exact same JSON body; injects keep_alive automatically.
    """
    body = await request.json()
    return await _enqueue_and_stream("/api/generate", body)


@app.post(
    "/api/chat",
    summary="Chat completion (queued + model-affinity scheduled)",
    response_description="NDJSON stream from Ollama /api/chat",
)
async def chat(request: Request):
    """
    Drop-in replacement for `POST http://localhost:11434/api/chat`.
    """
    body = await request.json()
    return await _enqueue_and_stream("/api/chat", body)


# ── Observability routes ───────────────────────────────────────────────────────

@app.get("/health", summary="Liveness / readiness probe")
async def health():
    """
    Returns HTTP 200 when the scheduler is running.
    Used by ALB health checks and container orchestrators.
    """
    running_models = await ollama.get_running_models()
    return {
        "status":                "healthy",
        "queue_depth":           request_queue.size,
        "currently_loaded_model": request_queue.currently_loaded_model,
        "running_models_in_vram": [m.get("name") for m in running_models],
        "counters": {
            "total_enqueued":  request_queue.total_enqueued,
            "total_processed": request_queue.total_processed,
            "total_dropped":   request_queue.total_dropped,
        },
    }


@app.get("/metrics", summary="Latency statistics (pure execution, queue wait, tok/s)")
async def metrics(model: str = None):
    """
    Returns P50/P95/P99 execution latency and tokens-per-second aggregates.

    Optional query parameter `model` filters results to a single model name.
    Example: GET /metrics?model=llama3.1:8b
    """
    all_models_seen = latency_tracker.all_models()
    return {
        "summary":   latency_tracker.get_summary(model),
        "per_model": {
            m: latency_tracker.get_summary(m) for m in all_models_seen
        },
        "queue": {
            "depth":            request_queue.size,
            "capacity":         settings.MAX_QUEUE_SIZE,
            "currently_loaded": request_queue.currently_loaded_model,
            "batch_window":     settings.BATCH_WINDOW_SIZE,
        },
    }


@app.get("/queue/status", summary="Real-time queue depth and pending request list")
async def queue_status():
    return {
        "queue_depth":           request_queue.size,
        "batch_window_size":     settings.BATCH_WINDOW_SIZE,
        "currently_loaded_model": request_queue.currently_loaded_model,
        "pending_requests":      request_queue.get_queue_snapshot(),
    }


@app.get("/models", summary="Available and VRAM-resident models")
async def list_models():
    available = await ollama.list_models()
    running   = await ollama.get_running_models()
    return {
        "available_locally": [
            {"name": m.get("name"), "size_gb": round(m.get("size", 0) / 1e9, 1)}
            for m in available
        ],
        "loaded_in_vram": running,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = settings.HOST,
        port      = settings.PORT,
        log_level = settings.LOG_LEVEL.lower(),
        loop      = "uvloop",       # uvloop is ~2× faster than asyncio default
        # workers=1  intentional: scheduler state must not be forked
    )
