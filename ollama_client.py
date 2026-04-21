"""
ollama_client.py — Async HTTP client for the Ollama REST API.

Responsibilities
────────────────
• Maintain a single persistent httpx.AsyncClient (connection pool).
• list_models()         → parses `GET /api/tags`
• get_running_models()  → parses `GET /api/ps`  (models resident in VRAM)
• unload_model()        → fires `POST /api/generate` with keep_alive=0 to evict
• stream_generate()     → async generator that yields raw NDJSON bytes
• check_model_available() → confirms model exists locally before dispatching

Error handling is defensive: individual failures are logged and surfaced as
JSON error chunks rather than raising unhandled exceptions into the scheduler.
"""
import logging
from typing import AsyncGenerator, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Stateful async wrapper around the Ollama local REST API."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Return (or lazily create) the shared async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.OLLAMA_BASE_URL,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=settings.OLLAMA_REQUEST_TIMEOUT,
                    write=30.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("OllamaClient HTTP connection pool closed.")

    # ── Model discovery ───────────────────────────────────────────────────────

    async def list_models(self) -> list[dict]:
        """
        Return all locally available models (equivalent to `ollama list`).
        Returns [] on failure so callers can handle gracefully.
        """
        client = await self._get_client()
        try:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except httpx.HTTPError as exc:
            logger.error(f"list_models failed: {exc}")
            return []

    async def get_running_models(self) -> list[dict]:
        """
        Return models currently loaded in VRAM via `GET /api/ps`.
        Requires Ollama ≥ 0.1.24.
        """
        client = await self._get_client()
        try:
            resp = await client.get("/api/ps")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except httpx.HTTPError as exc:
            logger.warning(f"get_running_models failed (old Ollama?): {exc}")
            return []

    async def check_model_available(self, model_name: str) -> bool:
        """
        Confirm that `model_name` exists in the local Ollama library.
        Handles both bare names ("llama3") and tagged names ("llama3:8b").
        """
        models = await self.list_models()
        # Build sets for both full-name and base-name matching
        full_names = {m.get("name", "") for m in models}
        base_names = {m.get("name", "").split(":")[0] for m in models}
        check_base = model_name.split(":")[0]
        return model_name in full_names or check_base in base_names

    # ── VRAM management ───────────────────────────────────────────────────────

    async def unload_model(self, model_name: str) -> bool:
        """
        Force-evict `model_name` from VRAM.

        Mechanism: POST /api/generate with an empty prompt and keep_alive=0.
        Ollama interprets keep_alive=0 as "unload immediately after this call."
        This is the official supported eviction path as of Ollama 0.1.24+.

        Returns True if the call was accepted, False on error.
        """
        logger.info(f"🔄  Unloading {model_name} from VRAM (keep_alive=0) …")
        client = await self._get_client()
        try:
            resp = await client.post(
                "/api/generate",
                json={
                    "model":      model_name,
                    "prompt":     "",
                    "keep_alive": 0,
                    "stream":     False,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            logger.info(f"✅  {model_name} unloaded successfully.")
            return True
        except httpx.HTTPError as exc:
            # A 404 or connection error here usually means the model wasn't
            # loaded in the first place — still a safe outcome.
            logger.warning(f"Unload call for {model_name} returned: {exc} (may be benign)")
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    async def stream_generate(
        self,
        endpoint: str,
        payload: dict,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream a generate or chat request to Ollama.

        Yields raw bytes (newline-delimited JSON) as they arrive.
        Network/HTTP errors are converted to a JSON error chunk so that
        downstream consumers always receive valid NDJSON.

        Parameters
        ──────────
        endpoint : "/api/generate" or "/api/chat"
        payload  : complete request body dict (keep_alive already injected
                   by the scheduler before this call)
        """
        client = await self._get_client()
        try:
            async with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                async for raw_chunk in response.aiter_bytes():
                    if raw_chunk:
                        yield raw_chunk

        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:200]
            yield (
                f'{{"error":"Ollama HTTP {exc.response.status_code}: {body}"}}\n'
            ).encode()

        except httpx.RequestError as exc:
            yield (
                f'{{"error":"Ollama connection error: {exc}"}}\n'
            ).encode()


# Module-level singleton — imported by scheduler.py and main.py
ollama = OllamaClient()
