"""Time phrases and ages (when.py)."""
import time
from datetime import datetime

from holonomic import when

# A fixed morning: Friday 2026-10-09, 09:52 local time.
NOW = datetime(2026, 10, 9, 9, 52).timestamp()


def day(y, m, d, h=12):
    return datetime(y, m, d, h).timestamp()


def inside(text, ts):
    w = when.window(text, NOW)
    return w is not None and w[0] <= ts < w[1]


def test_ages():
    assert when.age(day(2026, 10, 9, 1), NOW) == "today"
    assert when.age(day(2026, 10, 8, 23), NOW) == "yesterday"
    assert when.age(day(2026, 10, 6), NOW) == "3 days ago"
    assert when.age(day(2026, 9, 18), NOW) == "3 weeks ago"
    assert when.age(day(2026, 5, 1), NOW) == "5 months ago"
    assert when.age(day(2023, 1, 1), NOW) == "3 years ago"


def test_windows():
    assert inside("our late coding session last night", day(2026, 10, 8, 23))
    assert inside("our late coding session last night", day(2026, 10, 9, 1))
    assert not inside("our late coding session last night", day(2026, 10, 6, 23))
    assert inside("what did we talk about yesterday?", day(2026, 10, 8, 15))
    assert not inside("what did we talk about yesterday?", day(2026, 10, 9, 8))
    assert inside("that joke I told you last week", day(2026, 10, 1))
    assert not inside("that joke I told you last week", day(2026, 8, 1))
    assert inside("three days ago you said", day(2026, 10, 6))
    assert inside("a couple of weeks ago", day(2026, 9, 25))
    assert inside("this morning", day(2026, 10, 9, 8))
    assert when.window("Tell me about Lisbon", NOW) is None
    assert when.window("I'll do it next week", NOW) is None
    assert when.span(day(2026, 10, 8, 0), day(2026, 10, 9, 0)) == "2026-10-08"
