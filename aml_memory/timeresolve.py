"""Rule-based annotation of relative time expressions with the dates they refer to.

"I went there yesterday" said on 8 May 2023 becomes "I went there yesterday (7 May 2023)".
Week- and weekday-level expressions keep an anchored relative form ("the week before 9 June 2023"),
day/month/year-level ones get the absolute value. No model calls; deterministic.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "couple of": 2, "a few": 3, "few": 3, "several": 3,
}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUM_RE = r"(?P<n>\d+|a couple of|couple of|a few|few|several|an|a|one|two|three|four|five|six|seven|eight|nine|ten)"


def _d(day: date) -> str:
    return f"{day.day} {day:%B %Y}"


def _month_shift(day: date, k: int) -> str:
    m = day.month - 1 + k
    return f"{date(day.year + m // 12, m % 12 + 1, 1):%B %Y}"


def _ago(n_word: str, unit: str, day: date) -> str:
    n = _NUM.get(n_word.lower()) if not n_word.isdigit() else int(n_word)
    vague = n_word.lower() in ("a couple of", "couple of", "a few", "few", "several")
    unit = unit.lower().rstrip("s")
    if unit == "day":
        return f"{n_word} days before {_d(day)}" if vague else _d(day - timedelta(days=n))
    if unit == "week":
        return f"{n_word} weeks before {_d(day)}" if vague else f"{n} week{'s' if n > 1 else ''} before {_d(day)}"
    if unit == "month":
        return f"{n_word} months before {day:%B %Y}" if vague else _month_shift(day, -n)
    if unit == "year":
        return f"{n_word} years before {day.year}" if vague else str(day.year - n)
    return ""


def _rules(day: date):
    wd = day.weekday()

    def last_weekday(name: str) -> str:
        return f"the {name.capitalize()} before {_d(day)}"

    def next_weekday(name: str) -> str:
        return f"the {name.capitalize()} after {_d(day)}"

    return [
        (r"\bthe day before yesterday\b", lambda m: _d(day - timedelta(days=2))),
        (r"\bthe day after tomorrow\b", lambda m: _d(day + timedelta(days=2))),
        (r"\byesterday\b", lambda m: _d(day - timedelta(days=1))),
        (r"\blast night\b", lambda m: _d(day - timedelta(days=1))),
        (r"\b(?:today|tonight|this morning|this afternoon|this evening)\b", lambda m: _d(day)),
        (r"\btomorrow\b", lambda m: _d(day + timedelta(days=1))),
        (rf"\b{_NUM_RE} (?P<u>days?|weeks?|months?|years?) ago\b", lambda m: _ago(m.group("n"), m.group("u"), day)),
        (r"\blast weekend\b", lambda m: f"the weekend before {_d(day)}"),
        (r"\b(?:this|next) weekend\b", lambda m: f"the weekend after {_d(day)}"),
        (r"\blast week\b", lambda m: f"the week before {_d(day)}"),
        (r"\bthis week\b", lambda m: f"the week of {_d(day)}"),
        (r"\bnext week\b", lambda m: f"the week after {_d(day)}"),
        (r"\blast month\b", lambda m: _month_shift(day, -1)),
        (r"\bthis month\b", lambda m: _month_shift(day, 0)),
        (r"\bnext month\b", lambda m: _month_shift(day, 1)),
        (r"\blast year\b", lambda m: str(day.year - 1)),
        (r"\bthis year\b", lambda m: str(day.year)),
        (r"\bnext year\b", lambda m: str(day.year + 1)),
        (rf"\blast (?P<w>{'|'.join(_WEEKDAYS)})\b", lambda m: last_weekday(m.group("w"))),
        (rf"\bnext (?P<w>{'|'.join(_WEEKDAYS)})\b", lambda m: next_weekday(m.group("w"))),
        (rf"\bthis past (?P<w>{'|'.join(_WEEKDAYS)})\b", lambda m: last_weekday(m.group("w"))),
    ], wd


def annotate(text: str, ts_ms: int | None) -> str:
    if ts_ms is None or not text:
        return text
    day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
    rules, _ = _rules(day)
    spans: list[tuple[int, int, str]] = []
    for pattern, fn in rules:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if any(s < m.end() and m.start() < e for s, e, _ in spans):
                continue  # already covered by a longer expression
            if text[m.end() : m.end() + 2].lstrip().startswith("("):
                continue  # already annotated
            note = fn(m)
            if note:
                spans.append((m.start(), m.end(), note))
    for s, e, note in sorted(spans, reverse=True):
        text = f"{text[:e]} ({note}){text[e:]}"
    return text
