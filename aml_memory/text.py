"""Text helpers: keyword tokenizer, chunking, display formatting."""
from __future__ import annotations

import re
from datetime import datetime, timezone

STOPWORDS = set(
    """a an the and or but if then else of to in on at by for with about against between into through
    during before after above below from up down out off over under again further once here there when
    where why how all any both each few more most other some such no nor not only own same so than too
    very can will just don should now is are was were be been being have has had having do does did
    doing i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
    she her hers herself it its itself they them their theirs themselves what which who whom this that
    these those am would could ought i'm you're he's she's it's we're they're i've you've we've
    they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't aren't
    wasn't weren't hasn't haven't hadn't doesn't don't didn't won't wouldn't shan't shouldn't can't
    cannot couldn't mustn't let's that's who's what's here's there's when's where's why's how's
    did does do also get got like really yeah oh ok okay""".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[㐀-鿿]+")
_CJK = re.compile(r"[㐀-鿿]")


def _stem(w: str) -> str:
    # Light suffix stripping so "running"/"runs" meet "run" in keyword search.
    for suf in ("ingly", "edly", "ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def keyword_tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _WORD.findall(text.lower()):
        if _CJK.match(tok):
            if len(tok) == 1:
                out.append(tok)
            else:
                out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
            continue
        tok = tok.replace("'", "")
        if tok in STOPWORDS or len(tok) < 2 and not tok.isdigit():
            continue
        out.append(_stem(tok))
    return out


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer to cut at a sentence or line boundary.
            cut = max(text.rfind("\n", start + size // 2, end), text.rfind(". ", start + size // 2, end))
            if cut > start:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def format_time(ts_ms: int | None) -> str:
    if ts_ms is None:
        return ""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M (%a)")


def iso_time(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def format_line(role: str, content: str, ts_ms: int | None) -> str:
    when = format_time(ts_ms)
    prefix = f"[{when}] " if when else ""
    return f"{prefix}{role}: {content}"
