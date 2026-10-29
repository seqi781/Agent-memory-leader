"""SQLite-backed memory store with hybrid (dense + keyword) retrieval, isolated per user_id."""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import config
from .text import format_line, iso_time, keyword_tokens

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    rid INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    msg_idx INTEGER NOT NULL,
    chunk_idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    ts INTEGER,
    content TEXT NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, rid);
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(user_id, session_id, rid);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    n INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(ukey, terms, content='', tokenize='unicode61');
"""


@dataclass
class Piece:
    msg_idx: int
    chunk_idx: int
    role: str
    ts: int | None
    content: str


def _ukey(user_id: str) -> str:
    return "u" + hashlib.sha1(user_id.encode()).hexdigest()[:24]


class Store:
    def __init__(self) -> None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(config.DB_PATH, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._versions: dict[str, int] = {}
        self._cache: OrderedDict[str, tuple[int, np.ndarray, np.ndarray]] = OrderedDict()

    # ---------- write ----------
    def has_request(self, request_id: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM requests WHERE request_id=?", (request_id,)).fetchone()
        return row is not None

    def add(self, request_id: str, user_id: str, session_id: str, pieces: list[Piece], vecs: np.ndarray) -> bool:
        """Insert one Add request atomically. Returns False if it was already stored (idempotent retry)."""
        ukey = _ukey(user_id)
        blobs = [v.astype(np.float16).tobytes() for v in vecs]
        with self._lock:
            cur = self._db.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                if cur.execute("SELECT 1 FROM requests WHERE request_id=?", (request_id,)).fetchone():
                    cur.execute("ROLLBACK")
                    return False
                for p, blob in zip(pieces, blobs):
                    cur.execute(
                        "INSERT INTO memories(user_id, session_id, request_id, msg_idx, chunk_idx, role, ts, content, vec)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (user_id, session_id, request_id, p.msg_idx, p.chunk_idx, p.role, p.ts, p.content, blob),
                    )
                    terms = " ".join(keyword_tokens(f"{p.role} {p.content}"))
                    cur.execute("INSERT INTO fts(rowid, ukey, terms) VALUES (?,?,?)", (cur.lastrowid, ukey, terms))
                cur.execute(
                    "INSERT INTO requests(request_id, user_id, session_id, n) VALUES (?,?,?,?)",
                    (request_id, user_id, session_id, len(pieces)),
                )
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise
            self._versions[user_id] = self._versions.get(user_id, 0) + 1
        return True

    # ---------- read ----------
    def _user_vectors(self, user_id: str) -> tuple[np.ndarray, np.ndarray]:
        version = self._versions.get(user_id, 0)
        hit = self._cache.get(user_id)
        if hit and hit[0] == version:
            self._cache.move_to_end(user_id)
            return hit[1], hit[2]
        rows = self._db.execute("SELECT rid, vec FROM memories WHERE user_id=? ORDER BY rid", (user_id,)).fetchall()
        if rows:
            rids = np.array([r[0] for r in rows], dtype=np.int64)
            mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(len(rows), -1).astype(np.float32)
        else:
            rids, mat = np.zeros(0, dtype=np.int64), np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        self._cache[user_id] = (version, rids, mat)
        while len(self._cache) > config.VECTOR_CACHE_USERS:
            self._cache.popitem(last=False)
        return rids, mat

    def _keyword_search(self, user_id: str, query: str, limit: int) -> list[int]:
        terms = list(dict.fromkeys(keyword_tokens(query)))[:64]
        if not terms:
            return []
        match = f'ukey:{_ukey(user_id)} AND ({" OR ".join(chr(34) + t + chr(34) for t in terms)})'
        rows = self._db.execute(
            "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts, 0.0, 1.0) LIMIT ?", (match, limit)
        ).fetchall()
        return [r[0] for r in rows]

    def search(self, user_id: str, query_vecs: np.ndarray, keyword_query: str, top_k: int) -> list[dict]:
        if top_k <= 0:
            return []
        with self._lock:
            rids, mat = self._user_vectors(user_id)
            if len(rids) == 0:
                return []
            n = config.CANDIDATES
            # Dense: best similarity across the query variants.
            sims = (mat @ query_vecs.T).max(axis=1)
            dense_order = np.argsort(-sims)[:n]
            dense = [int(rids[i]) for i in dense_order]
            keyword = self._keyword_search(user_id, keyword_query, n)

            fused: dict[int, float] = {}
            for rank, rid in enumerate(dense):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
            for rank, rid in enumerate(keyword):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
            ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
            return [self._render(user_id, rid, score) for rid, score in ranked]

    def _render(self, user_id: str, rid: int, score: float) -> dict:
        row = self._db.execute(
            "SELECT session_id, role, ts, content FROM memories WHERE rid=? AND user_id=?", (rid, user_id)
        ).fetchone()
        session_id, role, ts, content = row
        lines = [format_line(role, content, ts)]
        w = config.NEIGHBOR_WINDOW
        if w > 0 and len(content) <= config.NEIGHBOR_MAX_CHARS:
            before = self._db.execute(
                "SELECT role, ts, content FROM memories WHERE user_id=? AND session_id=? AND rid<? ORDER BY rid DESC LIMIT ?",
                (user_id, session_id, rid, w),
            ).fetchall()
            after = self._db.execute(
                "SELECT role, ts, content FROM memories WHERE user_id=? AND session_id=? AND rid>? ORDER BY rid LIMIT ?",
                (user_id, session_id, rid, w),
            ).fetchall()
            ctx = lambda r: format_line(r[0], r[2][: config.NEIGHBOR_MAX_CHARS], r[1])  # noqa: E731
            lines = [ctx(r) for r in reversed(before)] + [">> " + lines[0]] + [ctx(r) for r in after]
        item = {"id": f"m{rid}", "content": "\n".join(lines), "score": round(float(score), 6)}
        created = iso_time(ts)
        if created:
            item["created_at"] = created
        return item

    def stats(self) -> dict:
        with self._lock:
            mem = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            users = self._db.execute("SELECT COUNT(DISTINCT user_id) FROM requests").fetchone()[0]
        return {"memories": mem, "users": users}
