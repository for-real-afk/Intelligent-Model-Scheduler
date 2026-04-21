import logging
import time
from typing import AsyncGenerator, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

# TTL cache for model availability — avoids a GET /api/tags on every request
_model_cache: set[str] = set()
_cache_ts: float = 0.0


class OllamaClient:
    """Stateful async wrapper around the Ollama local REST API."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
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

    async def list_models(self) -> list[dict]:
        client = await self._get_client()
        try:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except httpx.HTTPError as exc:
            logger.error(f"list_models failed: {exc}")
            return []

    async def get_running_models(self) -> list[dict]:
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
        Confirm model exists locally. Result is cached for 60 s to avoid
        a GET /api/tags round-trip on every single request.
        """
        global _model_cache, _cache_ts
        if time.monotonic() - _cache_ts > 60.0:
            models = await self.list_models()
            _model_cache = {m.get("name", "") for m in models}
            _model_cache |= {m.get("name", "").split(":")[0] for m in models}
            _cache_ts = time.monotonic()
            logger.debug(f"Model cache refreshed: {_model_cache}")
        check = model_name.split(":")[0]
        return model_name in _model_cache or check in _model_cache

    async def unload_model(self, model_name: str) -> bool:
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
            logger.warning(f"Unload call for {model_name} returned: {exc} (may be benign)")
            return False

    async def stream_generate(
        self,
        endpoint: str,
        payload: dict,
    ) -> AsyncGenerator[bytes, None]:
        client = await self._get_client()
        try:
            async with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                async for raw_chunk in response.aiter_bytes():
                    if raw_chunk:
                        yield raw_chunk
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:200]
            yield (f'{{"error":"Ollama HTTP {exc.response.status_code}: {body}"}}\n').encode()
        except httpx.RequestError as exc:
            yield (f'{{"error":"Ollama connection error: {exc}"}}\n').encode()


# Module-level singleton
ollama = OllamaClient()