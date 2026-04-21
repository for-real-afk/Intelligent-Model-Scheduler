"""
config.py — Centralised settings for the Ollama Model Scheduler.

All values can be overridden via environment variables prefixed with
SCHEDULER_  (e.g.  SCHEDULER_BATCH_WINDOW_SIZE=20).
A .env file in the working directory is also supported.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Ollama backend ────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_REQUEST_TIMEOUT: float = 300.0   # seconds per streaming request

    # ── Batch-window scheduling ───────────────────────────────────────────────
    BATCH_WINDOW_SIZE: int = 10             # max requests per batch
    # How long to wait collecting same-model requests before issuing the batch.
    # A small value (200 ms) lets bursts of identical-model calls group together
    # without introducing noticeable latency for single callers.
    BATCH_COLLECT_TIMEOUT_S: float = 0.20

    # ── Ollama keep_alive values ──────────────────────────────────────────────
    # Active: keep the model in VRAM while the queue is draining.
    KEEP_ALIVE_ACTIVE: str = "5m"
    # Force-unload: sent before switching to a heavy model to guarantee VRAM headroom.
    KEEP_ALIVE_UNLOAD: str = "0"

    # ── Queue back-pressure ───────────────────────────────────────────────────
    MAX_QUEUE_SIZE: int = 500
    # Maximum total wall-clock seconds a request may sit in queue + execution
    # before the StreamingResponse gives up and sends a timeout error to the client.
    REQUEST_TIMEOUT_S: float = 600.0

    # ── VRAM budget ───────────────────────────────────────────────────────────
    # Conservative ceiling on an A10G (24 GB).  Leaves ~2 GB for CUDA overhead
    # and the Ollama daemon itself.
    VRAM_BUDGET_GB: float = 22.0

    # ── FastAPI / Uvicorn ─────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    LOG_LEVEL: str = "INFO"

    model_config = {"env_prefix": "SCHEDULER_", "env_file": ".env"}


settings = Settings()

# ---------------------------------------------------------------------------
# Approximate VRAM consumption (GB, 4-bit quantised) keyed by parameter count.
# Used to decide whether an explicit model-unload is needed before switching.
# ---------------------------------------------------------------------------
MODEL_VRAM_MAP: dict[str, float] = {
    "1b":  1.0,  "2b":  1.5,  "3b":  2.0,  "4b":  3.0,
    "7b":  4.5,  "8b":  5.5,  "9b":  6.0,
    "13b": 8.5,  "14b": 9.0,
    "20b": 13.0, "22b": 14.0, "26b": 17.0, "27b": 17.5,
    "30b": 19.0, "32b": 20.0, "34b": 21.5, "35b": 22.0,
    "70b": 42.0, "72b": 44.0,
}
