from datetime import timedelta

from lib.temporal import Timer


def test_elapsed_measures_the_span_since_construction() -> None:
    reading = [5.0]
    timer = Timer(lambda: reading[0])

    reading[0] = 7.5
    assert timer.elapsed() == timedelta(seconds=2.5)

    reading[0] = 9.0
    assert timer.elapsed() == timedelta(seconds=4.0)
