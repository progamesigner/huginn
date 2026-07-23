from datetime import UTC, timedelta

from lib.temporal import ClockManager


def test_now_is_timezone_aware_utc() -> None:
    now = ClockManager().now()

    assert now.tzinfo is UTC
    assert now.utcoffset() == timedelta(0)


def test_timer_is_backed_by_a_monotonic_clock() -> None:
    timer = ClockManager().timer()

    assert timer.elapsed() >= timedelta(0)
