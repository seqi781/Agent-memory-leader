# Agent-memory-leader

A long-term memory service for the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/) (AML) Cycle 2 challenge, Textual track, Open-source Methods division.

It implements the AML Add / Search contract: Add stores conversation turns, and Search returns ranked memory evidence. The platform generates answers and scores them. Search never generates answers.

## Method (v0.2)

**Add**
- Each `user_id` gets its own SQLite database file, so isolation is physical and keyword statistics are computed per user.
- Each message becomes a memory record with its role, timestamp, session, and position. Messages longer than 1,500 characters are split into overlapping chunks.
- **Relative time resolution:** relative time expressions are annotated in place with the dates they refer to, computed from the message timestamp. It is rule-based and deterministic; no model is called.
  - `yesterday` → `yesterday (7 May 2023)`
  - `last year` → `last year (2022)`
  - `next month` → `next month (June 2023)`
  - Week- and weekday-level expressions keep an anchored relative form: `last week` → `last week (the week before 9 June 2023)`, `last Friday` → `last Friday (the Friday before 15 July 2023)`.
- Text is indexed for BM25 (SQLite FTS5), using stemmed English terms and CJK character bigrams.
- Add is synchronous. It returns `200` only after the write has committed and is searchable.
- Add is idempotent on `request_id`, so platform retries never duplicate memories.

**Search**
- BM25 ranks the user's turns against the question. When `options` are present, a second query that includes them is fused in with Reciprocal Rank Fusion.
- Each hit is expanded into a **session window**: the hit plus up to four adjacent turns on each side (at most 800 characters per side) from the same session. Overlapping or adjacent windows are merged.
- Windows are returned in relevance order. Each starts with its absolute date, for example `[Date: 8 May 2023 (Monday), 13:56]`, followed by `speaker: text` lines.
- Evidence is packed into a budget of about **12k tokens**. If the user's history is smaller than that, the remaining budget is filled with whole sessions, most recent first.
- Local tests on LoCoMo and LongMemEval-S found that a focused 12k context beats larger ones (24k–48k): evidence recall stays high, and the answer model is distracted less.

**Models (academic rules)**

| Component | Model |
|---|---|
| Embedding | none |
| LLM | none |
| Reranker | none |

No model is called at Add or Search time. If an embedding or LLM step is ever added, it will use `text-embedding-v4` / `gpt-4o-mini` as the rules require.

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
cp .env.example .env   # set AML_SERVICE_KEY
.venv/bin/uvicorn aml_memory.main:app --host 127.0.0.1 --port 8000 --workers 3
.venv/bin/python scripts/contract_test.py http://127.0.0.1:8000 "$AML_SERVICE_KEY"
```

Production deployment uses `deploy/aml-memory.service` (systemd) behind `deploy/Caddyfile` (automatic HTTPS with a Let's Encrypt IP certificate).

## Local evaluation

`scripts/eval_local.py` ingests LoCoMo or LongMemEval-S through the API and searches every question. It reports evidence recall at several context budgets. With `--llm`, it also answers and judges each question using the official AML prompts from [AML-memory/agent-memory-leaderboard](https://github.com/AML-memory/agent-memory-leaderboard), through any OpenAI-compatible endpoint.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CONTEXT_TOKENS` | 12000 | evidence budget per Search |
| `NEIGHBOR_TURNS` / `NEIGHBOR_CHARS` | 4 / 800 | session-window expansion around each hit |
| `CANDIDATES` | 400 | BM25 hits considered per query variant |
| `RESOLVE_TIME` | 1 | relative time annotation on/off |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` | 1500 / 200 | long-message chunking |

## Changelog

- **v0.2**: switched to per-user BM25 with session windows and relative-time resolution, using a 12k-token evidence budget. Removed the embedding step.
- **v0.1**: hybrid dense (`text-embedding-v4`) + BM25 retrieval.

## Attribution

This is original work for the AML 2026 challenge. It uses FastAPI and SQLite FTS5. Reciprocal Rank Fusion follows Cormack et al. (SIGIR 2009). The evaluation prompts come from the AML public repository.
