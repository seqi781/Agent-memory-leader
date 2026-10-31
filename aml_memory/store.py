"""Per-user SQLite memory store with BM25 retrieval and session-window evidence packing.

Each user_id gets its own database file, so isolation is physical, keyword statistics are
computed per user, and writes for different users never contend.
"""
from __future__ import annotations

import hashlib
import heapq
import logging
import re
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from . import config, rerank
from .notes import ROLE as NOTE_ROLE
from .text import STOPWORDS, est_tokens, format_time, iso_time, keyword_tokens

log = logging.getLogger("aml_memory.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    rid INTEGER PRIMARY KEY,
    sess_no INTEGER NOT NULL,
    pos INTEGER NOT NULL,
    role TEXT NOT NULL,
    ts INTEGER,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_pos ON turns(sess_no, pos);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    sess_no INTEGER NOT NULL UNIQUE,
    first_ts INTEGER
);
CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, n INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS vecs (rid INTEGER PRIMARY KEY, v BLOB NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(terms, content='', tokenize='unicode61 remove_diacritics 2');
"""

_CHUNK_NO = re.compile(r"chunk[-_:]?(\d+)\s*$")


@dataclass
class Piece:
    msg_idx: int
    chunk_idx: int
    role: str
    ts: int | None
    content: str


@dataclass
class Turn:
    rid: int
    sess_no: int
    pos: int
    role: str
    ts: int | None
    content: str


@dataclass
class Window:
    sess_no: int
    turns: list[Turn] = field(default_factory=list)

    @property
    def lo(self) -> int:
        return self.turns[0].pos

    @property
    def hi(self) -> int:
        return self.turns[-1].pos


class UserDB:
    def __init__(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA busy_timeout=60000")  # several worker processes may share a user file
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA cache_size=-1024")  # 1 MB page cache per open user file
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def close(self) -> None:
        with self.lock:
            self.db.close()


def _user_path(user_id: str):
    h = hashlib.sha1(user_id.encode()).hexdigest()
    return config.DATA_DIR / "users" / h[:2] / f"{h}.db"


class Store:
    def __init__(self) -> None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._open: OrderedDict[str, UserDB] = OrderedDict()
        self._lock = threading.Lock()

    def _user(self, user_id: str, create: bool) -> UserDB | None:
        with self._lock:
            udb = self._open.get(user_id)
            if udb is not None:
                self._open.move_to_end(user_id)
                return udb
            path = _user_path(user_id)
            if not create and not path.exists():
                return None
            udb = UserDB(path)
            self._open[user_id] = udb
            while len(self._open) > config.OPEN_USER_DBS:
                _, old = self._open.popitem(last=False)
                old.close()
            return udb

    def exists(self, user_id: str) -> bool:
        return user_id in self._open or _user_path(user_id).exists()

    # ---------- write ----------
    def has_request(self, user_id: str, request_id: str) -> bool:
        udb = self._user(user_id, create=False)
        if udb is None:
            return False
        with udb.lock:
            return udb.db.execute("SELECT 1 FROM requests WHERE request_id=?", (request_id,)).fetchone() is not None

    def add(
        self, request_id: str, user_id: str, session_id: str, pieces: list[Piece], vectors: np.ndarray | None = None
    ) -> bool:
        """Insert one Add request atomically. Returns False if it was already stored (idempotent retry)."""
        m = _CHUNK_NO.search(request_id)
        chunk_no = int(m.group(1)) if m else None
        udb = self._user(user_id, create=True)
        with udb.lock:
            cur = udb.db.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                if cur.execute("SELECT 1 FROM requests WHERE request_id=?", (request_id,)).fetchone():
                    cur.execute("ROLLBACK")
                    return False
                row = cur.execute("SELECT sess_no FROM sessions WHERE session_id=?", (session_id,)).fetchone()
                if row:
                    sess_no = row[0]
                else:
                    sess_no = cur.execute("SELECT COALESCE(MAX(sess_no), -1) + 1 FROM sessions").fetchone()[0]
                    first_ts = next((p.ts for p in pieces if p.ts is not None), None)
                    cur.execute("INSERT INTO sessions VALUES (?,?,?)", (session_id, sess_no, first_ts))
                # Order inside a session: platform chunk number if present, else arrival order.
                if chunk_no is None:
                    chunk_no = cur.execute(
                        "SELECT COALESCE(MAX(pos) / 10000000, -1) + 1 FROM turns WHERE sess_no=?", (sess_no,)
                    ).fetchone()[0]
                for i, p in enumerate(pieces):
                    pos = chunk_no * 10_000_000 + p.msg_idx * 1000 + min(p.chunk_idx, 999)
                    cur.execute(
                        "INSERT INTO turns(sess_no, pos, role, ts, content) VALUES (?,?,?,?,?)",
                        (sess_no, pos, p.role, p.ts, p.content),
                    )
                    terms = " ".join(keyword_tokens(f"{p.role} {p.content}"))
                    rid = cur.lastrowid
                    cur.execute("INSERT INTO fts(rowid, terms) VALUES (?,?)", (rid, terms))
                    if vectors is not None:
                        cur.execute("INSERT INTO vecs VALUES (?,?)", (rid, vectors[i].tobytes()))
                cur.execute("INSERT INTO requests VALUES (?,?)", (request_id, len(pieces)))
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise
        return True

    # ---------- read ----------
    def search(
        self,
        user_id: str,
        query: str,
        options: list[str],
        top_k: int,
        qvec: np.ndarray | None = None,
        extra: list[str] | None = None,
        xvec: np.ndarray | None = None,
        ovecs: np.ndarray | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        udb = self._user(user_id, create=False)
        if udb is None:
            return []
        with udb.lock:
            return _search(udb.db, query, options, top_k, qvec, extra, xvec, ovecs)

    def stats(self) -> dict:
        users = list((config.DATA_DIR / "users").glob("*/*.db")) if (config.DATA_DIR / "users").exists() else []
        return {"users": len(users)}


# ---------- retrieval ----------
def _bm25(db: sqlite3.Connection, text: str, limit: int) -> list[int]:
    terms = list(dict.fromkeys(keyword_tokens(text)))[:96]
    if not terms:
        return []
    match = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
    rows = db.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?", (match, limit)).fetchall()
    return [r[0] for r in rows]


def _dense(db: sqlite3.Connection, qvec: np.ndarray, limit: int) -> list[int]:
    """Nearest turns by cosine similarity. Vectors are scanned in blocks to keep memory flat."""
    q = qvec.astype(np.float32)
    best: list[tuple[float, int]] = []
    cur = db.execute("SELECT rid, v FROM vecs")
    while True:
        rows = cur.fetchmany(4096)
        if not rows:
            break
        mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(len(rows), -1)
        sims = mat.astype(np.float32) @ q
        k = min(limit, len(rows))
        idx = np.argpartition(-sims, k - 1)[:k]
        best = heapq.nlargest(limit, best + [(float(sims[i]), rows[i][0]) for i in idx])
    return [rid for _, rid in best]


def _rank_hits(
    db: sqlite3.Connection,
    query: str,
    options: list[str],
    qvec: np.ndarray | None = None,
    extra: list[str] | None = None,
    xvec: np.ndarray | None = None,
    ovecs: np.ndarray | None = None,
) -> list[int]:
    n = config.CANDIDATES
    if options and config.OPTION_MERGE:
        return _option_merge(db, query, options, qvec, ovecs)
    lists = [(_bm25(db, query, n), 1.0)]
    if options:
        lists.append((_bm25(db, query + "\n" + "\n".join(options), n), 1.0))
    if qvec is not None and config.DENSE_CANDIDATES > 0:
        lists.append((_dense(db, qvec, config.DENSE_CANDIDATES), config.DENSE_WEIGHT))
    for q in extra or []:
        lists.append((_bm25(db, q, n), config.EXPAND_WEIGHT))
    if xvec is not None and config.DENSE_CANDIDATES > 0:
        for v in xvec:
            lists.append((_dense(db, v, config.DENSE_CANDIDATES), config.EXPAND_WEIGHT * config.DENSE_WEIGHT))
    if len(lists) == 1:
        return lists[0][0]
    fused: dict[int, float] = {}
    for lst, weight in lists:
        for rank, rid in enumerate(lst):
            fused[rid] = fused.get(rid, 0.0) + weight / (config.RRF_K + rank + 1)
    return [rid for rid, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


def option_unique_words(options: list[str]) -> list[str]:
    """Words of each option that no other option uses (the part that tells the options apart)."""
    words = [re.findall(r"[a-z0-9']+", o.lower()) for o in options]
    sets = [{w for w in ws if len(w) > 2 and w not in STOPWORDS} for ws in words]
    out = []
    for i, ws in enumerate(words):
        others = set().union(*(x for j, x in enumerate(sets) if j != i))
        out.append(" ".join(w for w in dict.fromkeys(ws) if w in sets[i] and w not in others))
    return out


def _option_merge(
    db: sqlite3.Connection, query: str, options: list[str], qvec: np.ndarray | None, ovecs: np.ndarray | None
) -> list[int]:
    """One group per option: BM25(question + option), BM25(question + the option's own words) and
    dense(question + option), plus the question-only BM25 and dense lists. Inside a group a turn
    takes its best rank over the lists; across groups it takes its best group rank. Ties go to
    the summed reciprocal rank over all lists."""
    n, m = config.CANDIDATES, config.OPTION_LIST
    shared = [_bm25(db, query, n)]
    if qvec is not None and config.DENSE_CANDIDATES > 0:
        shared.append(_dense(db, qvec, config.DENSE_CANDIDATES))
    uniq = option_unique_words(options)
    every: list[list[int]] = list(shared)
    groups: list[list[list[int]]] = []
    for i, opt in enumerate(options):
        lists = [_bm25(db, query + "\n" + opt, n)[:m]]
        if uniq[i]:
            lists.append(_bm25(db, query + "\n" + uniq[i], n)[:m])
        if ovecs is not None and i < len(ovecs) and config.DENSE_CANDIDATES > 0:
            lists.append(_dense(db, ovecs[i], config.DENSE_CANDIDATES)[:m])
        every += lists
        groups.append(lists + shared)
    tie: dict[int, float] = {}
    for lst in every:
        for rank, rid in enumerate(lst):
            tie[rid] = tie.get(rid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
    best: dict[int, int] = {}
    for lists in groups:
        inside: dict[int, int] = {}
        for lst in lists:
            for rank, rid in enumerate(lst):
                if rank < inside.get(rid, 1 << 30):
                    inside[rid] = rank
        order = sorted(inside, key=lambda r: (inside[r], -tie[r]))
        for rank, rid in enumerate(order):
            if rank < best.get(rid, 1 << 30):
                best[rid] = rank
    return sorted(best, key=lambda r: (best[r], -tie[r]))


def _rerank(db: sqlite3.Connection, query: str, options: list[str], ranked: list[int]) -> list[int]:
    """Re-score the top BM25 candidates with the cross-encoder; the tail keeps BM25 order."""
    head, tail = ranked[: config.RERANK_TOP], ranked[config.RERANK_TOP :]
    docs, keep = [], []
    for rid in head:
        t = _turn(db, rid)
        if t is None:
            continue
        text = f"{t.role}: {t.content[: config.RERANK_CHARS]}"
        if config.RERANK_CONTEXT_CHARS > 0 and t.role != NOTE_ROLE:
            prev = _neighbors(db, t, True, 1)
            if prev:
                text = f"{prev[0].role}: {prev[0].content[: config.RERANK_CONTEXT_CHARS]}\n{text}"
        docs.append(text)
        keep.append(rid)
    q = query if not options else query + "\n" + "\n".join(options)
    try:
        sc = rerank.scores(q, docs)
    except Exception:  # never fail a Search because of the reranker: keep the fused order
        log.exception("rerank failed; using fused order")
        return ranked
    by_ce = [rid for _, rid in sorted(zip(sc, keep), key=lambda x: -x[0])]
    if config.RERANK_BLEND:
        fused: dict[int, float] = {}
        for lst in (keep, by_ce):
            for rank, rid in enumerate(lst):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        by_ce = [rid for rid, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
    return by_ce + tail


def _turn(db: sqlite3.Connection, rid: int) -> Turn | None:
    row = db.execute("SELECT rid, sess_no, pos, role, ts, content FROM turns WHERE rid=?", (rid,)).fetchone()
    return Turn(*row) if row else None


def _neighbors(db: sqlite3.Connection, t: Turn, before: bool, n: int) -> list[Turn]:
    # Nearest first on both sides; memory notes are never pulled in as neighbours.
    if before:
        sql = "SELECT rid, sess_no, pos, role, ts, content FROM turns WHERE sess_no=? AND pos<? AND role!=? ORDER BY pos DESC LIMIT ?"
    else:
        sql = "SELECT rid, sess_no, pos, role, ts, content FROM turns WHERE sess_no=? AND pos>? AND role!=? ORDER BY pos LIMIT ?"
    return [Turn(*r) for r in db.execute(sql, (t.sess_no, t.pos, NOTE_ROLE, n)).fetchall()]


def _expand(db: sqlite3.Connection, hit: Turn) -> list[Turn]:
    """Hit turn plus adjacent turns of the same session, up to a character allowance per side."""
    turns = [hit]
    if hit.role == NOTE_ROLE:
        return turns
    for before in (True, False):
        used = 0
        for nb in _neighbors(db, hit, before, config.NEIGHBOR_TURNS):
            if used + len(nb.content) > config.NEIGHBOR_CHARS and used > 0:
                break
            used += len(nb.content)
            if before:
                turns.insert(0, nb)
            else:
                turns.append(nb)
            if used > config.NEIGHBOR_CHARS:
                break
    return turns


def _search(
    db: sqlite3.Connection,
    query: str,
    options: list[str],
    top_k: int,
    qvec: np.ndarray | None = None,
    extra: list[str] | None = None,
    xvec: np.ndarray | None = None,
    ovecs: np.ndarray | None = None,
) -> list[dict]:
    budget = config.CONTEXT_TOKENS
    windows: list[Window] = []
    used = 0

    def add_turns(sess_no: int, turns: list[Turn]) -> int:
        """Merge turns into an overlapping/adjacent window of the session, or open a new one."""
        for w in windows:
            if w.sess_no == sess_no and _adjacent(db, w, turns):
                known = {t.rid for t in w.turns}
                new = [t for t in turns if t.rid not in known]
                w.turns = sorted(w.turns + new, key=lambda t: t.pos)
                return sum(est_tokens(t.content) + 4 for t in new)
        if len(windows) >= top_k:
            return -1
        windows.append(Window(sess_no, sorted(turns, key=lambda t: t.pos)))
        return sum(est_tokens(t.content) + 4 for t in turns) + 16

    covered: set[int] = set()
    ranked = _rank_hits(db, query, options, qvec, extra, xvec, ovecs)
    if rerank.enabled() and ranked and not (options and config.OPTION_MERGE and not config.OPTION_RERANK):
        ranked = _rerank(db, query, options, ranked)
    for rid in ranked:
        if used >= budget:
            break
        if rid in covered:
            continue
        hit = _turn(db, rid)
        if hit is None:
            continue
        turns = _expand(db, hit)
        turns = _fit(turns, hit, budget - used)
        cost = add_turns(hit.sess_no, turns)
        if cost < 0:
            break
        used += cost
        covered.update(t.rid for t in turns)

    # Fill any remaining budget with whole sessions, most recent first, so small histories are
    # returned in full and large ones still get recent context.
    if used < budget * 0.9 and len(windows) < top_k:
        for sess_no, in db.execute("SELECT sess_no FROM sessions ORDER BY COALESCE(first_ts, 0) DESC, sess_no DESC"):
            if used >= budget or len(windows) >= top_k:
                break
            rows = db.execute(
                "SELECT rid, sess_no, pos, role, ts, content FROM turns WHERE sess_no=? AND role!=? ORDER BY pos",
                (sess_no, NOTE_ROLE),
            ).fetchall()
            rest = [Turn(*r) for r in rows if r[0] not in covered]
            if not rest:
                continue
            chunk: list[Turn] = []
            for t in rest:
                c = est_tokens(t.content) + 4
                if used + c > budget:
                    break
                chunk.append(t)
                used += c
            if chunk:
                windows.append(Window(sess_no, chunk))
                covered.update(t.rid for t in chunk)

    n = len(windows)
    return [_render(w, rank, n) for rank, w in enumerate(windows[:top_k])]


def _adjacent(db: sqlite3.Connection, w: Window, turns: list[Turn]) -> bool:
    """True when no stored turn lies between the window and the new turns (gap-free in the session)."""
    if turns[0].pos > w.hi:
        lo, hi = w.hi, turns[0].pos
    elif turns[-1].pos < w.lo:
        lo, hi = turns[-1].pos, w.lo
    else:
        return True
    row = db.execute(
        "SELECT 1 FROM turns WHERE sess_no=? AND pos>? AND pos<? AND role!=? LIMIT 1", (w.sess_no, lo, hi, NOTE_ROLE)
    ).fetchone()
    return row is None


def _fit(turns: list[Turn], hit: Turn, remaining: int) -> list[Turn]:
    """Drop neighbours (furthest first) and finally truncate the hit so the window fits the budget."""
    turns = list(turns)
    while len(turns) > 1 and sum(est_tokens(t.content) + 4 for t in turns) > remaining:
        far = max((t for t in turns if t.rid != hit.rid), key=lambda t: abs(t.pos - hit.pos))
        turns.remove(far)
    if sum(est_tokens(t.content) for t in turns) > remaining and remaining > 64:
        keep = max(256, remaining * 3)
        turns = [Turn(hit.rid, hit.sess_no, hit.pos, hit.role, hit.ts, hit.content[:keep])]
    return turns


def _render(w: Window, rank: int, total: int) -> dict:
    lines = []
    first_ts = next((t.ts for t in w.turns if t.ts is not None), None)
    if first_ts is not None:
        lines.append(f"[Date: {format_time(first_ts)}]")
    last_ts = first_ts
    for t in w.turns:
        prefix = ""
        if t.ts is not None and last_ts is not None and abs(t.ts - last_ts) >= 60_000:
            prefix = f"[{format_time(t.ts)}] "
        if t.ts is not None:
            last_ts = t.ts
        lines.append(f"{prefix}{t.role}: {t.content}")
    item = {
        "id": f"s{w.sess_no}-{w.lo}-{w.hi}",
        "content": "\n".join(lines),
        "score": round(1.0 - rank / max(1, total), 6),
    }
    created = iso_time(first_ts)
    if created:
        item["created_at"] = created
    return item
