"""Write-time memory notes: gpt-4o-mini (the LLM the open-source division allows) rewrites each
Add request into short, self-contained notes (facts, preferences, events with dates, changes).

Notes are stored next to the raw turns and indexed the same way, so a question phrased differently
from the conversation can still reach the evidence. Raw turns are always kept; if the LLM call
fails, the Add is stored without notes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import threading
import time

import httpx

from . import config

log = logging.getLogger("aml_memory.notes")

ROLE = "memory note"

PROMPT = """You maintain long-term memory for a conversational assistant. Below is an excerpt of a \
conversation{date}. Write memory notes that would help answer questions about it later.

Rules:
- One fact per note, one self-contained sentence, third person, naming who it is about \
(use names when known, otherwise "the user" / "the assistant").
- Cover: personal facts, preferences, likes and dislikes, opinions, habits, goals and plans, \
events and when they happened, people and relationships, possessions, places, numbers and amounts, \
and specific things the assistant recommended or provided.
- When something changes earlier information (moved, changed job, new count, no longer likes...), \
say what changed.
- Turn relative times into absolute dates using the excerpt date (e.g. "on 7 May 2023", \
"in the week before 9 June 2023").
- Skip greetings, filler, and generic advice.
- At most {max_notes} notes. Return JSON: {{"notes": ["...", "..."]}}. Use {{"notes": []}} if nothing is worth keeping.

Conversation:
{text}"""

_client: httpx.Client | None = None
_client_lock = threading.Lock()
_sem = threading.BoundedSemaphore(max(1, config.NOTES_CONCURRENCY))
_usage = {"calls": 0, "prompt": 0, "completion": 0}
_usage_lock = threading.Lock()
_disk: sqlite3.Connection | None = None
_disk_lock = threading.Lock()


def enabled() -> bool:
    return bool(config.NOTES and enabled_llm())


def enabled_llm() -> bool:
    return bool(config.NOTES_MODEL and config.NOTES_API_KEY)


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                base_url=config.NOTES_BASE_URL.rstrip("/"),
                headers={"Authorization": f"Bearer {config.NOTES_API_KEY}"},
                timeout=config.NOTES_TIMEOUT,
                limits=httpx.Limits(max_connections=config.NOTES_CONCURRENCY * 2),
            )
        return _client


def _cache_db() -> sqlite3.Connection | None:
    global _disk
    if not config.NOTES_CACHE:
        return None
    with _disk_lock:
        if _disk is None:
            _disk = sqlite3.connect(config.NOTES_CACHE, check_same_thread=False, isolation_level=None)
            _disk.execute("PRAGMA journal_mode=WAL")
            _disk.execute("PRAGMA busy_timeout=60000")
            _disk.execute("CREATE TABLE IF NOT EXISTS notes (h TEXT PRIMARY KEY, v TEXT NOT NULL)")
        return _disk


def _parse(content: str) -> list[str]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{") :]
    try:
        data = json.loads(content)
    except ValueError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(content[start : end + 1])
        except ValueError:
            return []
    notes = data.get("notes") if isinstance(data, dict) else data
    if not isinstance(notes, list):
        return []
    out = []
    for n in notes:
        if isinstance(n, str) and n.strip():
            out.append(" ".join(n.split())[:600])
    return out[: config.NOTES_MAX]


def _call(text: str, date: str) -> list[str]:
    prompt = PROMPT.format(date=f" from {date}" if date else "", max_notes=config.NOTES_MAX, text=text)
    key = hashlib.sha1(f"{config.NOTES_MODEL}|{prompt}".encode()).hexdigest()
    disk = _cache_db()
    if disk is not None:
        with _disk_lock:
            row = disk.execute("SELECT v FROM notes WHERE h=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
    payload = {
        "model": config.NOTES_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": config.NOTES_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    last = ""
    for attempt in range(config.NOTES_RETRIES):
        try:
            with _sem:
                r = _http().post("/chat/completions", json=payload)
            if r.status_code == 200:
                body = r.json()
                usage = body.get("usage") or {}
                with _usage_lock:
                    _usage["calls"] += 1
                    _usage["prompt"] += int(usage.get("prompt_tokens") or 0)
                    _usage["completion"] += int(usage.get("completion_tokens") or 0)
                    if _usage["calls"] % 500 == 0:
                        log.info("notes usage: %s", _usage)
                notes = _parse(body["choices"][0]["message"].get("content") or "")
                if disk is not None:
                    with _disk_lock:
                        disk.execute("INSERT OR REPLACE INTO notes VALUES (?,?)", (key, json.dumps(notes)))
                return notes
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (408, 409, 429) and r.status_code < 500:
                break
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(20.0, 2.0**attempt) * (0.5 + random.random()))
    log.warning("notes failed: %s", last)
    return []


def extract(lines: list[str], date: str) -> list[str]:
    """Notes for one Add request. `lines` are "role: text" strings in order. Never raises."""
    segments, cur, size = [], [], 0
    for line in lines:
        line = line[: config.NOTES_SEGMENT_CHARS]
        if cur and size + len(line) > config.NOTES_SEGMENT_CHARS:
            segments.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        segments.append(cur)
    notes: list[str] = []
    for seg in segments[: config.NOTES_MAX_SEGMENTS]:
        try:
            notes.extend(_call("\n".join(seg), date))
        except Exception as e:  # never fail an Add because of notes
            log.warning("notes error: %s", e)
    return notes


def usage() -> dict:
    with _usage_lock:
        return dict(_usage)


EXPAND_PROMPT = """A user asked an assistant the request below. The assistant has a long memory of past \
conversations with this user. Write {n} short search queries that would find the personal information \
in that memory needed to answer well: the user's relevant preferences, likes and dislikes, background, \
constraints, plans, past events, or people involved. Be specific to the request; do not answer it.
Return JSON: {{"queries": ["...", "..."]}}

Request: {query}"""


def expand(query: str) -> list[str]:
    """Extra search queries for a request (search-time query rewriting). Never raises."""
    prompt = EXPAND_PROMPT.format(n=config.EXPAND_N, query=query[:4000])
    key = hashlib.sha1(f"x|{config.NOTES_MODEL}|{prompt}".encode()).hexdigest()
    disk = _cache_db()
    if disk is not None:
        with _disk_lock:
            row = disk.execute("SELECT v FROM notes WHERE h=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
    payload = {
        "model": config.NOTES_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(3):
        try:
            with _sem:
                r = _http().post("/chat/completions", json=payload, timeout=config.EXPAND_TIMEOUT)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"].get("content") or ""
                try:
                    data = json.loads(content[content.find("{") : content.rfind("}") + 1])
                except ValueError:
                    data = {}
                qs = [" ".join(q.split())[:300] for q in (data.get("queries") or []) if isinstance(q, str) and q.strip()]
                qs = qs[: config.EXPAND_N]
                if disk is not None:
                    with _disk_lock:
                        disk.execute("INSERT OR REPLACE INTO notes VALUES (?,?)", (key, json.dumps(qs)))
                return qs
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            log.warning("expand error: %s", e)
        time.sleep(1.0 + attempt)
    return []
