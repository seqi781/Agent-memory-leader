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


# Storage
DATA_DIR = Path(os.environ.get("AML_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "memory.db"

# Service auth: when set, requests must carry it as Bearer / Token / X-Api-Key.
SERVICE_KEY = os.environ.get("AML_SERVICE_KEY", "")

# Embedding (academic division: text-embedding-v4 only)
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "dashscope")  # dashscope | hash (local tests only)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_REGION = os.environ.get("DASHSCOPE_REGION", "intl")
EMBED_BASE_URL = os.environ.get(
    "EMBED_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if DASHSCOPE_REGION == "cn"
    else "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-v4")
EMBED_DIM = _int("EMBED_DIM", 1024)
EMBED_BATCH = _int("EMBED_BATCH", 10)  # text-embedding-v4 accepts at most 10 inputs per call
EMBED_CONCURRENCY = _int("EMBED_CONCURRENCY", 8)

# Chunking: long messages are split so each piece stays well inside the embedding limit.
CHUNK_CHARS = _int("CHUNK_CHARS", 1500)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)

# Retrieval
CANDIDATES = _int("CANDIDATES", 200)  # per retriever, before fusion
RRF_K = _int("RRF_K", 60)
NEIGHBOR_WINDOW = _int("NEIGHBOR_WINDOW", 1)  # adjacent turns added around short hits
NEIGHBOR_MAX_CHARS = _int("NEIGHBOR_MAX_CHARS", 600)
VECTOR_CACHE_USERS = _int("VECTOR_CACHE_USERS", 32)
