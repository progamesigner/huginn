from datetime import UTC, datetime, timedelta

from app.core.application import App
from app.core.providers.temporal import ClockProvider
from lib.temporal import ClockContract, ClockManager
from tests.helpers.clock import FakeClock


def test_provider_binds_clock_manager() -> None:
    container = App().build(ClockProvider())

    assert isinstance(container.resolve(ClockContract), ClockManager)


def test_fake_clock_swaps_in_via_container_use() -> None:
    container = App().build(ClockProvider())
    moment = datetime(2026, 1, 1, tzinfo=UTC)

    with container.use(ClockContract, FakeClock(now=moment)):
        assert container.resolve(ClockContract).now() == moment


def test_fake_clock_advance_drives_now_and_timer() -> None:
    clock = FakeClock(now=datetime(2026, 1, 1, tzinfo=UTC))
    timer = clock.timer()

    clock.advance(timedelta(seconds=5))

    assert clock.now() == datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)
    assert timer.elapsed() == timedelta(seconds=5)
