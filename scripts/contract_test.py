"""Check a running service against the AML Add/Search contract.

Usage: python scripts/contract_test.py https://host [service_key]
"""
from __future__ import annotations

import sys
import time
import uuid

import httpx

base = sys.argv[1].rstrip("/")
key = sys.argv[2] if len(sys.argv) > 2 else ""
headers = {"Authorization": f"Bearer {key}"} if key else {}
client = httpx.Client(base_url=base, headers=headers, timeout=120)
run = uuid.uuid4().hex[:8]
failures = 0


def check(cond: bool, msg: str) -> None:
    global failures
    print(("PASS " if cond else "FAIL ") + msg)
    failures += 0 if cond else 1


r = httpx.get(base + "/health", timeout=30)
check(200 <= r.status_code < 300, f"health unauthenticated -> {r.status_code}")

user_a, user_b = f"eval:{run}:locomo:conv-0", f"eval:{run}:locomo:conv-1"
conv = [
    ("Caroline", "I went to the LGBTQ support group yesterday and it was so powerful.", 1683536400000),
    ("Melanie", "That's great! I painted a sunrise over the lake last week.", 1683536460000),
    ("Caroline", "I'm planning to study counseling and mental health next year.", 1683536520000),
    ("Melanie", "My kids love camping; we went to the beach in July.", 1686214800000),
]
add_req = {
    "request_id": f"eval:{run}:locomo_refined:conv-0:chunk-0",
    "messages": [{"role": r_, "timestamp": ts, "content": c} for r_, c, ts in conv],
    "user_id": user_a,
    "session_id": f"eval:{run}:sample:0",
}
t = time.time()
r = client.post("/v1/memories/add", json=add_req)
body = r.json()
check(r.status_code == 200, f"add -> {r.status_code} in {time.time() - t:.1f}s")
check(body.get("success") is True, "add success is boolean true")
for k in ("request_id", "user_id", "session_id"):
    check(body.get(k) == add_req[k], f"add echoes {k}")

r = client.post("/v1/memories/add", json=add_req)
check(r.status_code == 200 and r.json().get("success") is True, "add retry with same request_id is idempotent")

other = {**add_req, "request_id": f"eval:{run}:x:conv-1:chunk-0", "user_id": user_b,
         "messages": [{"role": "user", "content": "The secret password of user B is pineapple."}]}
check(client.post("/v1/memories/add", json=other).status_code == 200, "add for second user")

q = {"query": "What did Caroline plan to study?", "user_id": user_a, "top_k": 100}
r = client.post("/v1/memories/search", json=q)
data = r.json().get("data")
check(r.status_code == 200 and isinstance(data, list), "search returns {data: [...]}")
check(len(data) <= 100, f"search count {len(data)} <= top_k")
check(all(isinstance(d.get("id"), str) and d["id"] and isinstance(d.get("content"), str) and d["content"] for d in data),
      "items have non-empty id and content")
check(bool(data) and "counseling" in data[0]["content"], "top hit contains the answer evidence")
check(not any("pineapple" in d["content"] for d in data), "no leakage across user_id")
print("   top1:", data[0]["content"].replace("\n", " | ")[:200] if data else None)

r = client.post("/v1/memories/search", json={**q, "top_k": 2,
                "options": ["A. Counseling", "B. Astronomy"]})
check(r.status_code == 200 and len(r.json()["data"]) <= 2, "search honours small top_k with options")
r = client.post("/v1/memories/search", json={"query": "anything", "user_id": f"eval:{run}:empty", "top_k": 100})
check(r.status_code == 200 and r.json() == {"data": []}, "unknown user returns data: []")
r = client.post("/v1/memories/add", json={"request_id": "x", "user_id": "u", "session_id": "s", "messages": []})
check(r.status_code in (400, 422), f"invalid add rejected with {r.status_code}")

print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
sys.exit(1 if failures else 0)
