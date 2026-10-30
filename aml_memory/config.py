"""Runtime settings, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# Storage: one SQLite file per user_id under DATA_DIR/users/
DATA_DIR = Path(os.environ.get("AML_DATA_DIR", ROOT / "data"))
OPEN_USER_DBS = _int("OPEN_USER_DBS", 64)

# Service auth: when set, requests must carry it as Bearer / Token / X-Api-Key.
SERVICE_KEY = os.environ.get("AML_SERVICE_KEY", "")

# Chunking: long messages are split into overlapping pieces.
CHUNK_CHARS = _int("CHUNK_CHARS", 1500)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)

# Retrieval
CANDIDATES = _int("CANDIDATES", 400)  # BM25 hits considered per query variant
RRF_K = _int("RRF_K", 60)
NEIGHBOR_TURNS = _int("NEIGHBOR_TURNS", 4)  # adjacent turns added on each side of a hit
NEIGHBOR_CHARS = _int("NEIGHBOR_CHARS", 800)  # character allowance per side
CONTEXT_TOKENS = _int("CONTEXT_TOKENS", 16000)  # evidence budget per Search (platform cap is ~117k)
RESOLVE_TIME = _int("RESOLVE_TIME", 1)  # annotate relative time expressions with absolute dates

# Cross-encoder reranker over the top BM25 candidates (set RERANK_MODEL= to turn it off).
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
RERANK_TOP = _int("RERANK_TOP", 100)  # BM25 candidates re-scored
RERANK_CHARS = _int("RERANK_CHARS", 600)  # characters of each candidate shown to the reranker
RERANK_CONTEXT_CHARS = _int("RERANK_CONTEXT_CHARS", 200)  # characters of the preceding turn added as context
RERANK_BLEND = _int("RERANK_BLEND", 0)  # 0 = order by reranker score, 1 = RRF of BM25 and reranker ranks
RERANK_THREADS = _int("RERANK_THREADS", 2)
RERANK_BATCH = _int("RERANK_BATCH", 16)
RERANK_MAX_TOKENS = _int("RERANK_MAX_TOKENS", 256)  # query + candidate, truncated

# Dense retrieval with text-embedding-v4 (leave EMBED_MODEL empty to run keyword-only).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-v4")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
EMBED_DIM = _int("EMBED_DIM", 1024)
EMBED_BATCH = _int("EMBED_BATCH", 10)  # API maximum per request
EMBED_CONCURRENCY = _int("EMBED_CONCURRENCY", 8)  # in-flight API calls per worker process
EMBED_CHARS = _int("EMBED_CHARS", 2000)  # characters of each turn sent for embedding
EMBED_TIMEOUT = float(os.environ.get("EMBED_TIMEOUT", 30))
EMBED_RETRIES = _int("EMBED_RETRIES", 9)
EMBED_CACHE = os.environ.get("EMBED_CACHE", "")  # optional sqlite file caching vectors (local experiments)
DENSE_CANDIDATES = _int("DENSE_CANDIDATES", 200)  # nearest turns by cosine added to the candidate pool
DENSE_WEIGHT = float(os.environ.get("DENSE_WEIGHT", 1.0))  # weight of the dense list in rank fusion

# Write-time memory notes with gpt-4o-mini (NOTES=0 turns them off).
NOTES = _int("NOTES", 0)
NOTES_MODEL = os.environ.get("NOTES_MODEL", "openai/gpt-4o-mini")
NOTES_BASE_URL = os.environ.get("NOTES_BASE_URL", "https://openrouter.ai/api/v1")
NOTES_API_KEY = os.environ.get("NOTES_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
NOTES_CONCURRENCY = _int("NOTES_CONCURRENCY", 8)  # in-flight LLM calls per worker process
NOTES_TIMEOUT = float(os.environ.get("NOTES_TIMEOUT", 90))
NOTES_RETRIES = _int("NOTES_RETRIES", 6)
NOTES_MAX = _int("NOTES_MAX", 15)  # notes per LLM call
NOTES_MAX_TOKENS = _int("NOTES_MAX_TOKENS", 900)
NOTES_SEGMENT_CHARS = _int("NOTES_SEGMENT_CHARS", 24000)  # conversation text per LLM call
NOTES_MAX_SEGMENTS = _int("NOTES_MAX_SEGMENTS", 4)  # LLM calls per Add request at most
NOTES_CACHE = os.environ.get("NOTES_CACHE", "")  # optional sqlite cache (local experiments)

# Search-time query rewriting with the same LLM (QUERY_EXPAND=0 turns it off).
QUERY_EXPAND = _int("QUERY_EXPAND", 0)
EXPAND_N = _int("EXPAND_N", 3)
EXPAND_WEIGHT = float(os.environ.get("EXPAND_WEIGHT", 1.0))  # weight of each rewritten-query list in fusion
EXPAND_TIMEOUT = float(os.environ.get("EXPAND_TIMEOUT", 20))
