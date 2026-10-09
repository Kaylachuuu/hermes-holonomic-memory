"""Time in what she is told and what she is asked.

A recalled memory carries its date, but a model reading "2026-10-06" next to "last night" does not do the sum, so
each one also says how long ago it was.  And when a message names a time ("last week", "yesterday", "three days
ago"), recall keeps to the memories from then.  Local time throughout, as the user means it.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

DAY = 86400.0
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "a couple": 2, "a few": 3, "few": 3, "several": 4}
_N = r"(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|a couple(?: of)?|a few|few|several)"
_UNIT = {"day": 1, "week": 7, "month": 30, "year": 365}


def _part(hour: int) -> str:
    return "morning" if 5 <= hour < 12 else "afternoon" if hour < 17 else "evening" if hour < 22 else "night"


def age(ts: float, now: Optional[float] = None) -> str:
    """How long ago, the way a person says it: "this morning", "last night", "yesterday afternoon", "3 days ago".
    For today and yesterday the part of the day is said too, since "yesterday" alone covers a working day and the
    evening after it."""
    now = time.time() if now is None else now
    days = (_midnight(now) - _midnight(ts)).days
    hour = datetime.fromtimestamp(ts).hour
    if days <= 0:
        if hour < 5:
            return "last night" if datetime.fromtimestamp(now).hour < 12 else "early this morning"
        return {"morning": "this morning", "afternoon": "this afternoon", "evening": "this evening",
                "night": "tonight"}[_part(hour)]
    if days == 1:
        return "last night" if hour >= 22 or (hour >= 17 and datetime.fromtimestamp(now).hour < 12) \
            else f"yesterday {_part(hour)}"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    if days < 730:
        return f"{days // 30} months ago"
    return f"{days // 365} years ago"


def _midnight(ts: float) -> datetime:
    d = datetime.fromtimestamp(ts)
    return datetime(d.year, d.month, d.day)


def _num(word: str) -> int:
    word = word.lower()
    return int(word) if word.isdigit() else _NUMBERS.get(word, 1)


def window(text: str, now: Optional[float] = None) -> Optional[Tuple[float, float, str]]:
    """(start, end, the words that named it) when the message names a time in the past, else None.  The window
    is a little wider than the words, since people round: "last week" can be eight days ago."""
    now = time.time() if now is None else now
    t = (text or "").lower()
    today = _midnight(now)
    start_of = lambda days_back: (today - timedelta(days=days_back)).timestamp()      # noqa: E731
    m = re.search(r"\b(this morning|earlier today|today|tonight|this afternoon|this evening)\b", t)
    if m:
        return start_of(0), now, m.group(1)
    m = re.search(r"\b(last night|yesterday(?: (morning|afternoon|evening|night))?)\b", t)
    if m:
        y = today - timedelta(days=1)
        if m.group(1) == "last night":
            # Yesterday evening into the small hours of today.
            return (y + timedelta(hours=17)).timestamp(), (today + timedelta(hours=6)).timestamp(), m.group(1)
        hours = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 24), "night": (20, 30)}.get(m.group(2) or "")
        if hours:
            return (y + timedelta(hours=hours[0])).timestamp(), (y + timedelta(hours=hours[1])).timestamp(), m.group(1)
        return start_of(1), start_of(0), m.group(1)
    m = re.search(r"\b(the day before yesterday)\b", t)
    if m:
        return start_of(2), start_of(1), m.group(1)
    m = re.search(rf"\b({_N} (day|week|month|year)s? ago)\b", t)
    if m:
        n, unit = _num(m.group(2)), _UNIT[m.group(3)]
        back = n * unit
        slack = max(1, back // 4) if unit == 1 else max(unit // 2, back // 4)
        return start_of(back + slack), start_of(max(0, back - slack)) + DAY, m.group(1)
    m = re.search(r"\b(last|this|past) (week|month|year)\b", t)
    if m:
        which, unit = m.group(1), _UNIT[m.group(2)]
        if which == "last":
            return start_of(unit * 2 + 1), start_of(max(0, unit // 2)) + DAY, m.group(0)
        return start_of(unit), now, m.group(0)
    m = re.search(r"\b(the other day|recently|lately)\b", t)
    if m:
        return start_of(7), now, m.group(1)
    return None


def span(start: float, end: float) -> str:
    """The dates a window covers, for saying so."""
    a, b = time.strftime("%Y-%m-%d", time.localtime(start)), time.strftime("%Y-%m-%d", time.localtime(end - 1))
    return a if a == b else f"{a} to {b}"
