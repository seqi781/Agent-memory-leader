"""Dense embeddings with text-embedding-v4 (the model the open-source division requires), called
through an OpenAI-compatible endpoint.

Vectors are L2-normalised and stored as float16. Calls retry with backoff on rate limits and
server errors; callers fall back to keyword-only retrieval when the service stays unavailable.
An optional on-disk cache (EMBED_CACHE) lets local experiments re-run without paying again.
"""
from __future__ import annotations

import hashlib
import logging
import random
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from . import config

log = logging.getLogger("aml_memory.embed")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per API call is too much at Full scale


class EmbedError(Exception):
    pass


_client: httpx.Client | None = None
_client_lock = threading.Lock()
_sem = threading.BoundedSemaphore(max(1, config.EMBED_CONCURRENCY))
_pool = ThreadPoolExecutor(max_workers=max(1, config.EMBED_CONCURRENCY))
_qcache: OrderedDict[str, np.ndarray] = OrderedDict()
_qcache_lock = threading.Lock()
_usage = {"calls": 0, "tokens": 0}
_usage_lock = threading.Lock()
_disk: sqlite3.Connection | None = None
_disk_lock = threading.Lock()


def enabled() -> bool:
    return bool(config.EMBED_MODEL and config.EMBED_BASE_URL and config.EMBED_API_KEY)


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                base_url=config.EMBED_BASE_URL.rstrip("/"),
                headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
                timeout=config.EMBED_TIMEOUT,
                limits=httpx.Limits(max_connections=config.EMBED_CONCURRENCY * 2),
                # Hosts without working IPv6 stall on every new connection unless bound to IPv4.
                transport=httpx.HTTPTransport(
                    local_address="0.0.0.0", limits=httpx.Limits(max_connections=config.EMBED_CONCURRENCY * 2)
                ) if config.EMBED_IPV4 else None,
            )
        return _client


def _cache_db() -> sqlite3.Connection | None:
    global _disk
    if not config.EMBED_CACHE:
        return None
    with _disk_lock:
        if _disk is None:
            _disk = sqlite3.connect(config.EMBED_CACHE, check_same_thread=False, isolation_level=None)
            _disk.execute("PRAGMA journal_mode=WAL")
            _disk.execute("PRAGMA busy_timeout=60000")
            _disk.execute("CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, v BLOB NOT NULL)")
        return _disk


def _key(text: str) -> str:
    return hashlib.sha1(f"{config.EMBED_MODEL}|{config.EMBED_DIM}|{text}".encode()).hexdigest()


def _post(batch: list[str]) -> list[list[float]]:
    payload = {"model": config.EMBED_MODEL, "input": batch, "dimensions": config.EMBED_DIM, "encoding_format": "float"}
    last = ""
    for attempt in range(config.EMBED_RETRIES):
        try:
            with _sem:
                r = _http().post("/embeddings", json=payload)
            if r.status_code == 200:
                body = r.json()
                with _usage_lock:
                    _usage["calls"] += 1
                    _usage["tokens"] += int((body.get("usage") or {}).get("total_tokens") or 0)
                    if _usage["calls"] % 1000 == 0:
                        log.info("embedding usage: %d calls, %d tokens", _usage["calls"], _usage["tokens"])
                rows = sorted(body["data"], key=lambda d: d["index"])
                if len(rows) != len(batch):
                    raise EmbedError(f"expected {len(batch)} vectors, got {len(rows)}")
                return [d["embedding"] for d in rows]
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (408, 409, 429) and r.status_code < 500:
                raise EmbedError(last)  # bad request: retrying will not help
        except httpx.HTTPError as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))
    raise EmbedError(f"embedding failed after {config.EMBED_RETRIES} attempts: {last}")


def _normalise(rows: list[list[float]]) -> np.ndarray:
    m = np.asarray(rows, dtype=np.float32)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    return m.astype(np.float16)


def embed(texts: list[str]) -> np.ndarray:
    """(len(texts), EMBED_DIM) float16 matrix of unit vectors. Raises EmbedError."""
    texts = [(t or " ")[: config.EMBED_CHARS] for t in texts]
    out: list[np.ndarray | None] = [None] * len(texts)
    disk = _cache_db()
    if disk is not None:
        keys = [_key(t) for t in texts]
        with _disk_lock:
            for i, k in enumerate(keys):
                row = disk.execute("SELECT v FROM emb WHERE h=?", (k,)).fetchone()
                if row:
                    out[i] = np.frombuffer(row[0], dtype=np.float16)
    todo = [i for i, v in enumerate(out) if v is None]
    batches = [todo[i : i + config.EMBED_BATCH] for i in range(0, len(todo), config.EMBED_BATCH)]
    results = _pool.map(lambda b: _normalise(_post([texts[i] for i in b])), batches) if len(batches) > 1 else (
        [_normalise(_post([texts[i] for i in b])) for b in batches]
    )
    for b, mat in zip(batches, results):
        for i, vec in zip(b, mat):
            out[i] = vec
        if disk is not None:
            with _disk_lock:
                disk.executemany(
                    "INSERT OR REPLACE INTO emb VALUES (?,?)", [(_key(texts[i]), vec.tobytes()) for i, vec in zip(b, mat)]
                )
    if not out:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float16)
    return np.stack(out)


def embed_query(text: str) -> np.ndarray:
    with _qcache_lock:
        hit = _qcache.get(text)
        if hit is not None:
            _qcache.move_to_end(text)
            return hit
    vec = embed([text])[0]
    with _qcache_lock:
        _qcache[text] = vec
        while len(_qcache) > 2048:
            _qcache.popitem(last=False)
    return vec


def usage() -> dict:
    with _usage_lock:
        return dict(_usage)
