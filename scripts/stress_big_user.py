"""Stress test: write many LongMemEval haystacks into ONE user (BEAM-scale history), then time Search.

Usage: python scripts/stress_big_user.py lme_subset.jsonl http://127.0.0.1:8001 [key] [concurrency]
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

path, base = sys.argv[1], sys.argv[2]
key = sys.argv[3] if len(sys.argv) > 3 else ""
conc = int(sys.argv[4]) if len(sys.argv) > 4 else 16


async def main() -> None:
    user = f"stress:{uuid.uuid4().hex[:6]}"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    bodies, questions, chars = [], [], 0
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        x = json.loads(line)
        questions.append(x["question"])
        for k, (sid, date, sess) in enumerate(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"])):
            ts = int(datetime.strptime(date.split(" (")[0], "%Y/%m/%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            msgs = [{"role": t["role"], "timestamp": ts, "content": t["content"]} for t in sess if t.get("content", "").strip()]
            chars += sum(len(m["content"]) for m in msgs)
            for j in range(0, len(msgs), 20):
                bodies.append({"request_id": f"{user}:{x['question_id']}:{k}:chunk-{j // 20}", "messages": msgs[j : j + 20],
                               "user_id": user, "session_id": f"{user}:{x['question_id']}:{sid}"})
    print(f"{len(bodies)} Add requests, ~{chars / 4 / 1e6:.1f}M tokens into one user", flush=True)
    sem = asyncio.Semaphore(conc)
    lat: list[float] = []
    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=1800) as c:
        async def add(b):
            async with sem:
                t = time.time()
                r = await c.post("/v1/memories/add", json=b)
                r.raise_for_status()
                lat.append(time.time() - t)

        t0 = time.time()
        await asyncio.gather(*(add(b) for b in bodies))
        print(f"ingest {time.time() - t0:.0f}s, add p50 {statistics.median(lat):.2f}s max {max(lat):.2f}s", flush=True)

        slat: list[float] = []

        async def search(q):
            async with sem:
                t = time.time()
                r = await c.post("/v1/memories/search", json={"query": q, "user_id": user, "top_k": 100})
                r.raise_for_status()
                slat.append(time.time() - t)
                return r.json()["data"]

        t0 = time.time()
        res = await asyncio.gather(*(search(q) for q in questions))
        print(f"{len(questions)} searches in {time.time() - t0:.1f}s, p50 {statistics.median(slat):.2f}s max {max(slat):.2f}s, "
              f"items avg {sum(len(r) for r in res) / len(res):.0f}", flush=True)


asyncio.run(main())
