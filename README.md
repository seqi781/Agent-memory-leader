# Agent-memory-leader

A long-term memory service for the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/) (AML) Cycle 2 challenge, Textual track, Open-source Methods division.

It implements the AML Add / Search contract: Add stores conversation turns, and Search returns ranked memory evidence. The platform generates answers and scores them. Search never generates answers.

## Method (v0.4)

**Add**
- Each `user_id` gets its own SQLite database file, so isolation is physical and keyword statistics are computed per user.
- Each message becomes a memory record with its role, timestamp, session, and position. Messages longer than 1,500 characters are split into overlapping chunks.
- **Relative time resolution:** relative time expressions are annotated in place with the dates they refer to, computed from the message timestamp. It is rule-based and deterministic; no model is called.
  - `yesterday` → `yesterday (7 May 2023)`
  - `last year` → `last year (2022)`
  - `next month` → `next month (June 2023)`
  - Week- and weekday-level expressions keep an anchored relative form: `last week` → `last week (the week before 9 June 2023)`, `last Friday` → `last Friday (the Friday before 15 July 2023)`.
- Text is indexed for BM25 (SQLite FTS5), using stemmed English terms and CJK character bigrams.
- **Dense vectors:** each record (`role: text`, first 2,000 characters) is embedded with `text-embedding-v4` (1024 dimensions), L2-normalised, and stored as float16 in the same user database. Calls are batched (10 texts per request), run concurrently, and retry with backoff. If the embedding service stays unavailable, the Add is still stored and remains searchable by keyword.
- Add is synchronous. It returns `200` only after the write has committed and is searchable.
- Add is idempotent on `request_id`, so platform retries never duplicate memories.

**Search**
- **Hybrid retrieval:** BM25 ranks the user's turns against the question; when `options` are present, a second BM25 query that includes them is added. The question is also embedded with `text-embedding-v4`, and the 200 nearest records by cosine similarity form a dense list. The lists are fused with Reciprocal Rank Fusion. Dense retrieval finds paraphrased evidence that shares no keywords with the question. If the query embedding fails, Search falls back to BM25 only.
- **Cross-encoder reranking:** the top 100 fused candidates are re-scored by `ms-marco-MiniLM-L-6-v2`. It is a 22M-parameter ONNX model running on the service's own CPU (memory arena off, inputs capped at 256 tokens), so it adds no API cost. Each candidate is scored together with the preceding turn, so short replies keep their context. In local tests, evidence recall within a 12k budget rose from 83% to 96% on LongMemEval-S and from 92.5% to 95% on LoCoMo.
- Each hit is expanded into a **session window**: the hit plus up to four adjacent turns on each side (at most 800 characters per side) from the same session. Overlapping or adjacent windows are merged.
- Windows are returned in relevance order. Each starts with its absolute date, for example `[Date: 8 May 2023 (Monday), 13:56]`, followed by `speaker: text` lines.
- Evidence is packed into a budget of about **16k tokens**. If the user's history is smaller than that, the remaining budget is filled with whole sessions, most recent first.
- Local tests on LoCoMo and LongMemEval-S found that a focused context of 8k–16k beats larger ones (24k–48k): evidence recall stays high (97% at 16k with reranking), and the answer model is distracted less.

**Models (academic rules)**

| Component | Model |
|---|---|
| Embedding | `text-embedding-v4` (Alibaba Cloud DashScope, OpenAI-compatible API), 1024 dimensions |
| LLM | none in the submitted configuration |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (ONNX export `Xenova/ms-marco-MiniLM-L-6-v2`, Apache-2.0), local CPU |

The only hosted model is the embedding model. The code also contains two optional `gpt-4o-mini` steps (write-time memory notes, `NOTES=1`, and search-time query rewriting, `QUERY_EXPAND=1`). Both are **off** in the submitted configuration: in a local PersonaMem-v2 check (4 personas, 130 questions), notes changed 9 answers to right and 9 to wrong, so no net gain. On first start, the reranker is loaded from `models/`; if it is not there, it is downloaded from Hugging Face.

**Local results that motivated v0.4** (answer and judge model `qwen-flash`, AML prompts): on LoCoMo (2 conversations, 233 questions), adding dense retrieval raised evidence recall at 16k from 97.1% to 99.7% and answer accuracy from 81.5% to 82.8%; temporal questions rose from 82.5% to 85.7%. On PersonaMem-v2 (130 questions, `gpt-4o-mini` multiple choice), accuracy moved from 38.5% to 39.2%.

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

Production deployment uses systemd units `deploy/aml-memory-v4.service` (v0.4, port 8010) and `deploy/aml-memory.service` (v0.3, port 8000) behind `deploy/Caddyfile` (automatic HTTPS with a Let's Encrypt IP certificate). The registered endpoints are the root paths, served by v0.4; v0.3 stays reachable under `/v3` for reproduction.

## Local evaluation

`scripts/eval_local.py` ingests LoCoMo or LongMemEval-S through the API and searches every question. It reports evidence recall at several context budgets. With `--llm`, it also answers and judges each question using the official AML prompts from [AML-memory/agent-memory-leaderboard](https://github.com/AML-memory/agent-memory-leaderboard), through any OpenAI-compatible endpoint.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CONTEXT_TOKENS` | 16000 | evidence budget per Search |
| `NEIGHBOR_TURNS` / `NEIGHBOR_CHARS` | 4 / 800 | session-window expansion around each hit |
| `CANDIDATES` | 400 | BM25 hits considered per query variant |
| `RESOLVE_TIME` | 1 | relative time annotation on/off |
| `RERANK_MODEL` | Xenova/ms-marco-MiniLM-L-6-v2 | cross-encoder (empty string = off) |
| `RERANK_TOP` / `RERANK_CHARS` / `RERANK_CONTEXT_CHARS` | 100 / 600 / 200 | candidates re-scored and text shown to the reranker |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` | 1500 / 200 | long-message chunking |
| `EMBED_MODEL` / `EMBED_DIM` | text-embedding-v4 / 1024 | embedding model (empty string = keyword only) |
| `EMBED_BASE_URL` / `EMBED_API_KEY` | DashScope compatible-mode / `DASHSCOPE_API_KEY` | embedding endpoint |
| `EMBED_CONCURRENCY` / `EMBED_BATCH` | 8 / 10 | in-flight embedding calls per worker, texts per call |
| `DENSE_CANDIDATES` / `DENSE_WEIGHT` | 200 / 1.0 | nearest records added to fusion, weight of the dense list |
| `NOTES` / `QUERY_EXPAND` | 0 / 0 | optional `gpt-4o-mini` steps (off in the submission) |

## Changelog

- **v0.4**: hybrid retrieval. Every record is embedded with `text-embedding-v4` at Add time, and Search fuses a dense cosine list with the BM25 lists before cross-encoder reranking. Empty messages inside an Add are now skipped instead of rejecting the request.
- **v0.3**: added a local cross-encoder reranker over the BM25 candidates, widened session windows to 4 turns per side, and raised the evidence budget to 16k tokens.
- **v0.2**: switched to per-user BM25 with session windows and relative-time resolution, using a 12k-token evidence budget. Removed the embedding step.
- **v0.1**: hybrid dense (`text-embedding-v4`) + BM25 retrieval.

## Attribution

This is original work for the AML 2026 challenge. It uses FastAPI, SQLite FTS5, NumPy, the `text-embedding-v4` API, ONNX Runtime with Hugging Face tokenizers, and the MS MARCO MiniLM cross-encoder from sentence-transformers. Reciprocal Rank Fusion follows Cormack et al. (SIGIR 2009). The evaluation prompts come from the AML public repository.
