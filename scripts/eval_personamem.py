"""Local PersonaMem-v2 (32k, text) check: ingest each persona's chat history through Add, search
with each benchmark query, answer the official multiple-choice prompt with the retrieved memories,
and score the chosen letter.

The persona system message is not ingested (it states the preferences outright).

usage: eval_personamem.py --bench benchmark.csv --histories DIR --personas 8 --out DIR
       [--service URL] [--llm-base URL --llm-key-env NAME --model NAME]
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import collections
import csv
import json
import os
import random
import re
import time
from pathlib import Path

import httpx

RECALL_SUFFIX = (
    " Please recall my related preferences from our conversation history to give personalized responses."
)
MCQ_PROMPT = """Please choose the best answer from the following options:

{options}

Think step by step about which answer best fits the user's query and conversation context.
Provide your reasoning first, then give your final answer as 'Final Answer: [Letter]'"""
HF = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main/"


def query_text(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            return str(ast.literal_eval(raw).get("content", ""))
        except (ValueError, SyntaxError):
            pass
    return raw


def options_for(row: dict) -> tuple[list[str], str]:
    wrong = row["incorrect_answers"]
    try:
        wrong = json.loads(wrong)
    except ValueError:
        wrong = ast.literal_eval(wrong)
    opts = [row["correct_answer"]] + [str(w) for w in wrong]
    random.Random(f"{row['persona_id']}_{query_text(row['user_query'])}").shuffle(opts)
    letter = chr(65 + opts.index(row["correct_answer"]))
    return opts, letter


def final_letter(text: str) -> str:
    m = re.findall(r"Final Answer:\s*\**\s*\[?\(?([A-Z])\b", text or "")
    if m:
        return m[-1]
    m = re.findall(r"\b([A-D])\b", text or "")
    return m[-1] if m else ""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--histories", required=True)
    ap.add_argument("--personas", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--service", default="http://127.0.0.1:8001")
    ap.add_argument("--key", default="")
    ap.add_argument("--msgs-per-add", type=int, default=10)
    ap.add_argument("--no-options", action="store_true", help="do not send options with Search")
    ap.add_argument("--empty", action="store_true", help="baseline: answer without any memories")
    ap.add_argument("--llm-base", default="https://openrouter.ai/api/v1")
    ap.add_argument("--llm-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    csv.field_size_limit(10**9)
    rows = list(csv.DictReader(open(args.bench, encoding="utf-8")))
    by_p: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_p[r["persona_id"]].append(r)
    pids = sorted(by_p, key=int)
    random.Random(args.seed).shuffle(pids)
    pids = pids[: args.personas]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hdir = Path(args.histories)
    hdir.mkdir(parents=True, exist_ok=True)
    svc_headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    llm_key = os.environ[args.llm_key_env]
    sem = asyncio.Semaphore(args.concurrency)
    tag = f"pm{int(time.time())}"

    async with httpx.AsyncClient(timeout=1800) as client:
        # ---- ingest ----
        t0 = time.time()
        for pid in pids:
            link = by_p[pid][0]["chat_history_32k_link"]
            path = hdir / Path(link).name
            if not path.exists():
                r = await client.get(HF + link, follow_redirects=True)
                r.raise_for_status()
                path.write_bytes(r.content)
            hist = [m for m in json.loads(path.read_text())["chat_history"] if m.get("role") != "system"]
            user = f"{tag}-{pid}"
            chunks = [hist[i : i + args.msgs_per_add] for i in range(0, len(hist), args.msgs_per_add)]

            async def add(n: int, msgs: list[dict]) -> None:
                async with sem:
                    body = {
                        "request_id": f"{user}-chunk-{n}", "user_id": user, "session_id": f"{user}-s0",
                        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                    }
                    r = await client.post(args.service + "/v1/memories/add", json=body, headers=svc_headers)
                    r.raise_for_status()

            await asyncio.gather(*(add(n, c) for n, c in enumerate(chunks)))
        print(f"ingest {time.time() - t0:.0f}s for {len(pids)} personas")

        # ---- search + answer ----
        async def one(pid: str, row: dict) -> dict:
            q = query_text(row["user_query"])
            opts, gold = options_for(row)
            async with sem:
                body = {"user_id": f"{tag}-{pid}", "query": q, "top_k": 100}
                if not args.no_options:
                    body["options"] = opts
                s0 = time.time()
                r = await client.post(args.service + "/v1/memories/search", json=body, headers=svc_headers)
                r.raise_for_status()
                items = [] if args.empty else r.json()["data"]
                stime = time.time() - s0
                memory = "\n\n".join(f"[{i + 1}] {it['content']}" for i, it in enumerate(items))
                option_text = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
                messages = [
                    {"role": "system", "content": f"Relevant memories from previous conversations with this user:\n\n{memory}"},
                    {"role": "user", "content": q + RECALL_SUFFIX},
                    {"role": "system", "content": MCQ_PROMPT.format(options=option_text)},
                ]
                for attempt in range(5):
                    try:
                        rr = await client.post(
                            args.llm_base.rstrip("/") + "/chat/completions",
                            headers={"Authorization": f"Bearer {llm_key}"},
                            json={"model": args.model, "messages": messages, "temperature": 0},
                            timeout=120,
                        )
                        rr.raise_for_status()
                        text = rr.json()["choices"][0]["message"]["content"]
                        break
                    except Exception:  # noqa: BLE001
                        await asyncio.sleep(2 * (attempt + 1))
                else:
                    text = ""
            pred = final_letter(text)
            hit = row["preference"].lower()[:40] in memory.lower()
            return {
                "pid": pid, "q": q, "gold": gold, "pred": pred, "ok": pred == gold, "pref_type": row["pref_type"],
                "updated": row["updated"], "pref": row["preference"], "pref_in_ctx": hit, "search_s": round(stime, 2),
                "ctx_chars": len(memory),
            }

        recs = await asyncio.gather(*(one(pid, r) for pid in pids for r in by_p[pid]))
    with open(out / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = lambda rs: 100.0 * sum(r["ok"] for r in rs) / max(1, len(rs))  # noqa: E731
    print(f"ALL {len(recs):5d} acc {acc(recs):5.1f}  search avg {sum(r['search_s'] for r in recs) / len(recs):.2f}s")
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r["pref_type"]].append(r)
        groups["updated=" + r["updated"]].append(r)
    for k in sorted(groups):
        print(f"  {k:32s} {len(groups[k]):4d} acc {acc(groups[k]):5.1f}")


if __name__ == "__main__":
    asyncio.run(main())
