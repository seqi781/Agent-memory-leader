"""Local evaluation of a running memory service on LoCoMo or LongMemEval-S.

Ingests each conversation/haystack through the Add API, runs Search (top_k=100) for every
question, and reports evidence recall at several context budgets (free). With --llm, it also
answers and judges with the official AML prompts through an OpenAI-compatible endpoint.

  python scripts/eval_local.py --dataset locomo --data locomo10.json --pipeline AML_REPO --out DIR
  python scripts/eval_local.py --dataset lme --data lme_subset.jsonl --pipeline AML_REPO --out DIR
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aml_memory.text import est_tokens  # noqa: E402

BUDGETS = [8000, 16000, 32000, 64000, 100000]
LOCOMO_CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def load_pipeline(repo: Path, name: str):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location(f"{name}_pipeline", repo / f"data/{name}/pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


# ---------- dataset adapters: yield (user_key, sessions, questions) ----------
# sessions: list of (session_key, ts_ms, [ {role, content} ])
# questions: list of {question, gold, category, evidence: [text snippets]}
def locomo_samples(path: str):
    for idx, conv in enumerate(json.loads(Path(path).read_text())):
        c = conv["conversation"]
        sessions, n = [], 1
        while f"session_{n}" in c:
            ts = ms(datetime.strptime(c[f"session_{n}_date_time"].strip(), "%I:%M %p on %d %B, %Y"))
            turns = []
            for t in c[f"session_{n}"]:
                text = t["text"] + (f" [shares an image: {t['blip_caption']}]" if t.get("blip_caption") else "")
                turns.append({"role": t["speaker"], "content": text, "dia_id": t["dia_id"]})
            sessions.append((f"session-{n}", ts, turns))
            n += 1
        by_id = {t["dia_id"]: t["content"] for _, _, ts_ in sessions for t in ts_}
        qs = [
            {"question": q["question"], "gold": str(q.get("answer", "")), "category": LOCOMO_CATEGORY[q["category"]],
             "evidence": [by_id[e][:120] for e in q.get("evidence", []) if e in by_id]}
            for q in conv["qa"] if q.get("category") in LOCOMO_CATEGORY
        ]
        yield f"conv-{idx}", sessions, qs


def lme_samples(path: str):
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        x = json.loads(line)
        sessions, evidence = [], []
        for sid, date, sess in zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]):
            ts = ms(datetime.strptime(date.split(" (")[0] + " " + date.split(") ")[-1], "%Y/%m/%d %H:%M"))
            turns = [{"role": t["role"], "content": t["content"]} for t in sess if t.get("content", "").strip()]
            evidence += [t["content"][:120] for t in sess if t.get("has_answer")]
            sessions.append((sid, ts, turns))
        cat = x["question_type"] + ("_abs" if x["question_id"].endswith("_abs") else "")
        q = {"question": x["question"], "gold": str(x["answer"]), "category": cat, "evidence": evidence,
             "question_date": x.get("question_date")}
        yield x["question_id"], sessions, [q]


# ---------- runner ----------
async def ingest(svc: httpx.AsyncClient, user_id: str, run: str, sessions, sem: asyncio.Semaphore) -> None:
    async def one(skey, ts, turns):
        msgs = [{"role": t["role"], "timestamp": ts, "content": t["content"]} for t in turns]
        for j in range(0, len(msgs), 20):
            body = {"request_id": f"{user_id}:{skey}:chunk-{j // 20}", "messages": msgs[j : j + 20],
                    "user_id": user_id, "session_id": f"{user_id}:{skey}"}
            async with sem:
                r = await svc.post("/v1/memories/add", json=body)
            r.raise_for_status()

    # Sessions of one user are sent in order (the platform streams history chronologically).
    for s in sessions:
        if s[2]:
            await one(*s)


async def llm(client, base, key, model, prompt, sem) -> tuple[str, int]:
    async with sem:
        for attempt in range(6):
            try:
                r = await client.post(base.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {key}"},
                                      json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                            "temperature": 0, "max_tokens": 256}, timeout=300)
                if r.status_code == 200:
                    j = r.json()
                    return j["choices"][0]["message"]["content"].strip(), j.get("usage", {}).get("prompt_tokens", 0)
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["locomo", "lme"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--service", default="http://127.0.0.1:8001")
    ap.add_argument("--key", default="")
    ap.add_argument("--limit", type=int, default=0, help="max users (conversations / LME questions)")
    ap.add_argument("--max-questions", type=int, default=0, help="per-user question cap (locomo)")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--llm-base", default="https://openrouter.ai/api/v1")
    ap.add_argument("--llm-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--answer-model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge-model", default="openai/gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    pipe = load_pipeline(Path(args.pipeline), "locomo-refined" if args.dataset == "locomo" else "longmemeval-s")
    samples = locomo_samples(args.data) if args.dataset == "locomo" else lme_samples(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = uuid.uuid4().hex[:6]
    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    llm_key = os.environ.get(args.llm_key_env, "")
    sem = asyncio.Semaphore(args.concurrency)
    records: list[dict] = []
    prompt_tokens = 0
    t_ingest = t_search = 0.0

    async with httpx.AsyncClient(base_url=args.service, headers=headers, timeout=600) as svc, httpx.AsyncClient() as lc:
        for n_user, (ukey, sessions, qs) in enumerate(samples):
            if args.limit and n_user >= args.limit:
                break
            if args.max_questions:
                qs = qs[: args.max_questions]
            user_id = f"{args.dataset}:{run}:{ukey}"
            t0 = time.time()
            await ingest(svc, user_id, run, sessions, sem)
            t_ingest += time.time() - t0

            async def one(q: dict) -> dict:
                nonlocal prompt_tokens, t_search
                t1 = time.time()
                async with sem:
                    r = await svc.post("/v1/memories/search", json={"query": q["question"], "user_id": user_id, "top_k": 100})
                r.raise_for_status()
                t_search += time.time() - t1
                hits = [d["content"] for d in r.json()["data"]]
                cum, positions = 0, []
                cum_at = []
                for h in hits:
                    cum += est_tokens(h)
                    cum_at.append(cum)
                for e in q["evidence"]:
                    pos = next((cum_at[i] for i, h in enumerate(hits) if e in h), None)
                    positions.append(pos)
                rec = {"user": ukey, **{k: q[k] for k in ("question", "gold", "category")},
                       "evidence_token_pos": positions, "context_tokens": cum, "items": len(hits)}
                if args.llm:
                    prompt = pipe.render_answer_prompt({"question": q["question"], "retrieved_context": hits})
                    rec["generated_answer"], pt = await llm(lc, args.llm_base, llm_key, args.answer_model, prompt, sem)
                    jr, pt2 = await llm(lc, args.llm_base, llm_key, args.judge_model,
                                        pipe.render_accuracy_prompt({**q, "gold_answer": q["gold"]}, rec["generated_answer"]), sem)
                    prompt_tokens += pt + pt2
                    try:
                        rec["correct"] = pipe.parse_judge_label(jr) == "CORRECT"
                    except (ValueError, json.JSONDecodeError):
                        rec["correct"] = bool(re.search(r"\bCORRECT\b", jr)) and "WRONG" not in jr
                return rec

            records.extend(await asyncio.gather(*(one(q) for q in qs)))
            print(f"{ukey}: {len(qs)} q, ingest so far {t_ingest:.0f}s", flush=True)

    (out / "records.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    by = defaultdict(list)
    for r in records:
        by[r["category"]].append(r)
        by["ALL"].append(r)
    head = " ".join(f"R@{b // 1000}k".rjust(7) for b in BUDGETS)
    print(f"\n{'category':28} {'n':>4} {head} {'ctx':>7} {'acc':>6}")
    for cat, rs in sorted(by.items()):
        ev = [p for r in rs for p in r["evidence_token_pos"]]
        rec = " ".join(f"{100 * sum(1 for p in ev if p is not None and p <= b) / max(1, len(ev)):7.1f}" for b in BUDGETS)
        ctx = sum(r["context_tokens"] for r in rs) / len(rs)
        acc = f"{100 * sum(r.get('correct', False) for r in rs) / len(rs):6.1f}" if args.llm else "     -"
        print(f"{cat:28} {len(rs):4d} {rec} {ctx:7.0f} {acc}")
    print(f"\ningest {t_ingest:.0f}s, search total {t_search:.0f}s, LLM prompt tokens {prompt_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
