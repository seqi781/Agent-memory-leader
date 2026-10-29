"""Embedding client for text-embedding-v4 (DashScope OpenAI-compatible API)."""
from __future__ import annotations

import asyncio
import hashlib
import random
import re

import httpx
import numpy as np

from . import config

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class Embedder:
    def __init__(self) -> None:
        self.provider = config.EMBED_PROVIDER
        self.dim = config.EMBED_DIM
        self._sem = asyncio.Semaphore(config.EMBED_CONCURRENCY)
        self._client: httpx.AsyncClient | None = None
        if self.provider == "dashscope" and not config.DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY is not set (or use EMBED_PROVIDER=hash for local tests)")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=config.EMBED_BASE_URL,
                headers={"Authorization": f"Bearer {config.DASHSCOPE_API_KEY}"},
                timeout=httpx.Timeout(120.0, connect=15.0),
            )
        return self._client

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, one row per text."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches = [texts[i : i + config.EMBED_BATCH] for i in range(0, len(texts), config.EMBED_BATCH)]
        parts = await asyncio.gather(*(self._embed_batch(b) for b in batches))
        vecs = np.vstack(parts).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    async def _embed_batch(self, texts: list[str]) -> np.ndarray:
        texts = [t if t.strip() else "." for t in texts]
        if self.provider == "hash":
            return np.vstack([_hash_vector(t, self.dim) for t in texts])
        async with self._sem:
            delay = 1.0
            for attempt in range(8):
                try:
                    resp = await self._http().post(
                        "/embeddings",
                        json={
                            "model": config.EMBED_MODEL,
                            "input": texts,
                            "dimensions": self.dim,
                            "encoding_format": "float",
                        },
                    )
                except httpx.TransportError:
                    if attempt == 7:
                        raise
                else:
                    if resp.status_code == 200:
                        data = sorted(resp.json()["data"], key=lambda d: d["index"])
                        return np.array([d["embedding"] for d in data], dtype=np.float32)
                    if resp.status_code not in _RETRY_STATUS or attempt == 7:
                        raise RuntimeError(f"embedding API {resp.status_code}: {resp.text[:300]}")
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 30)
        raise RuntimeError("unreachable")


def _hash_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic bag-of-words vector. Only for offline tests, never for evaluation."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+|[一-鿿]", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 64) & 1 else -1.0
    return vec
