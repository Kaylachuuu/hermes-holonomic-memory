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
    assert when.age(day(2026, 10, 9, 1), NOW) == "last night"
    assert when.age(day(2026, 10, 9, 6), NOW) == "this morning"
    # Within the last few hours, roughly how long (2026-10-10: finer within a day).
    assert when.age(NOW - 5 * 60, NOW) == "a few minutes ago"
    assert when.age(NOW - 30 * 60, NOW) == "about half an hour ago"
    assert when.age(NOW - 70 * 60, NOW) == "about an hour ago"
    assert when.age(day(2026, 10, 9, 8), NOW) == "about two hours ago"
    assert when.age(NOW - 3 * 3600, NOW) == "about three hours ago"
    assert when.age(day(2026, 10, 8, 23), NOW) == "last night"
    assert when.age(day(2026, 10, 8, 19), NOW) == "last night"
    assert when.age(day(2026, 10, 8, 14), NOW) == "yesterday afternoon"
    assert when.age(day(2026, 10, 8, 19), datetime(2026, 10, 9, 15).timestamp()) == "yesterday evening"
    assert when.age(day(2026, 10, 6), NOW) == "3 days ago"
    assert when.age(day(2026, 9, 22), NOW) == "17 days ago"          # days, not "2 weeks", up to three weeks
    assert when.age(day(2026, 9, 18), NOW) == "3 weeks ago"
    assert when.age(day(2026, 5, 1), NOW) == "5 months ago"
    assert when.age(day(2023, 1, 1), NOW) == "3 years ago"


def test_windows():
    assert inside("what did I say an hour ago?", NOW - 3600) and not inside("an hour ago", NOW - 5 * 3600)
    assert inside("a few hours ago", NOW - 3 * 3600)
    assert inside("a few minutes ago", NOW - 5 * 60) and not inside("a few minutes ago", NOW - 3 * 3600)
    assert inside("our late coding session last night", day(2026, 10, 8, 23))
    assert inside("our late coding session last night", day(2026, 10, 9, 1))
    assert not inside("our late coding session last night", day(2026, 10, 6, 23))
    assert not inside("our late coding session last night", day(2026, 10, 8, 14))     # yesterday afternoon
    assert inside("yesterday afternoon", day(2026, 10, 8, 14)) and not inside("yesterday afternoon", day(2026, 10, 8, 20))
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
