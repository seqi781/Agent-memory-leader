# Agent-memory-leader

A long-term memory service for the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/) (AML) Cycle 2 challenge, Textual track, Open-source Methods division.

It implements the AML Add / Search contract: Add stores conversation turns, and Search returns ranked memory evidence. The platform generates answers and scores them. Search never generates answers.

## Method

**Add**
- Every message becomes a memory record. Each record keeps its role, its timestamp (Unix ms, per message or per request), its session, and its order.
- Messages longer than 1,500 characters are split into overlapping chunks.
- Each record is embedded with **`text-embedding-v4`** (1024 dimensions) and indexed for BM25 keyword search (SQLite FTS5).
- Add is synchronous. It returns `200` only after the write has been committed and is searchable.
- Add is idempotent on `request_id`, so platform retries never duplicate memories.

**Search**
- Dense retrieval: cosine similarity against the query. For multiple-choice questions, the query is also embedded together with `options`, and each memory keeps the better of the two similarities.
- Keyword retrieval: BM25 over stemmed terms (CJK text uses character bigrams).
- Fusion: Reciprocal Rank Fusion (k = 60) of the two candidate lists. Up to `top_k` items are returned in relevance order.
- Each hit is returned with its absolute date and weekday, `[2023-05-08 09:02 (Mon)] Caroline: ...`, to support temporal reasoning. Short hits also include their adjacent turns from the same session; the hit line is marked with `>>`.
- Isolation: retrieval is scoped strictly to `user_id`. No state is shared across users or samples.

**Models (academic rules)**

| Component | Model |
|---|---|
| Embedding | `text-embedding-v4` (Alibaba Cloud Model Studio) |
| LLM | none in the current version (any future LLM step will use `gpt-4o-mini`) |
| Reranker | none in the current version |

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/memories/add` | AML Add contract |
| `POST` | `/v1/memories/search` | AML Search contract, `{"data": [{id, content, score, created_at}]}` |
| `GET` | `/health` | unauthenticated |

Auth: when `AML_SERVICE_KEY` is set, requests must send `Authorization: Bearer <key>` (`Token <key>` and `X-Api-Key` are also accepted).

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # set DASHSCOPE_API_KEY, DASHSCOPE_REGION, AML_SERVICE_KEY
.venv/bin/uvicorn aml_memory.main:app --host 127.0.0.1 --port 8000
.venv/bin/python scripts/contract_test.py http://127.0.0.1:8000 "$AML_SERVICE_KEY"
```

To test offline without an embedding key, run with `EMBED_PROVIDER=hash`. This mode is for tests only and is never used for evaluation.

Production deployment uses `deploy/aml-memory.service` (systemd) behind `deploy/Caddyfile` (automatic HTTPS).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `EMBED_DIM` | 1024 | text-embedding-v4 output dimensions |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` | 1500 / 200 | long-message chunking |
| `CANDIDATES` | 200 | candidates per retriever before fusion |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant |
| `NEIGHBOR_WINDOW` | 1 | adjacent turns attached to short hits |

## Attribution

This is original work for the AML 2026 challenge. It uses FastAPI, SQLite FTS5, and NumPy. Reciprocal Rank Fusion follows Cormack et al. (SIGIR 2009).
