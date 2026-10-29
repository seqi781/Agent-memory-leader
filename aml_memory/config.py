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
OPEN_USER_DBS = _int("OPEN_USER_DBS", 256)

# Service auth: when set, requests must carry it as Bearer / Token / X-Api-Key.
SERVICE_KEY = os.environ.get("AML_SERVICE_KEY", "")

# Chunking: long messages are split into overlapping pieces.
CHUNK_CHARS = _int("CHUNK_CHARS", 1500)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)

# Retrieval
CANDIDATES = _int("CANDIDATES", 400)  # BM25 hits considered per query variant
RRF_K = _int("RRF_K", 60)
NEIGHBOR_TURNS = _int("NEIGHBOR_TURNS", 2)  # adjacent turns added on each side of a hit
NEIGHBOR_CHARS = _int("NEIGHBOR_CHARS", 800)  # character allowance per side
CONTEXT_TOKENS = _int("CONTEXT_TOKENS", 12000)  # evidence budget per Search (platform cap is ~117k)
RESOLVE_TIME = _int("RESOLVE_TIME", 1)  # annotate relative time expressions with absolute dates
