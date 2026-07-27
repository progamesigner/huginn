import asyncio
import logging
from datetime import timedelta
from inspect import CORO_CLOSED, getcoroutinestate

import pytest

from lib.concurrency import ConcurrencyError, TaskSupervisor, TaskSupervisorContract
from lib.container import Container


async def test_spawn_runs_and_reaps_the_task() -> None:
    supervisor = TaskSupervisor()
    done = asyncio.Event()

    async def work() -> None:
        done.set()

    supervisor.spawn(work())
    assert len(supervisor.tasks) == 1

    async with asyncio.timeout(1):
        await done.wait()
    await asyncio.sleep(0)

    assert supervisor.tasks == set()


async def test_background_exception_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    supervisor = TaskSupervisor()

    async def boom() -> None:
        raise RuntimeError('kaboom')

    supervisor.spawn(boom())
    task = next(iter(supervisor.tasks))
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert 'Background task failed' in caplog.text
    assert supervisor.tasks == set()


async def test_drain_returns_immediately_with_nothing_in_flight() -> None:
    assert await TaskSupervisor().drain(timedelta(seconds=1))


async def test_drain_waits_for_the_work_to_finish() -> None:
    supervisor = TaskSupervisor()
    done = asyncio.Event()

    async def work() -> None:
        await asyncio.sleep(0.01)
        done.set()

    supervisor.spawn(work())

    assert await supervisor.drain(timedelta(seconds=1))
    assert done.is_set()
    assert supervisor.tasks == set()


async def test_drain_cancels_what_outlives_the_window_and_waits_for_the_cleanup() -> (
    None
):
    supervisor = TaskSupervisor()
    cleaned = asyncio.Event()

    async def work() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.set()

    supervisor.spawn(work())

    assert not await supervisor.drain(timedelta(milliseconds=10))
    assert cleaned.is_set()
    assert supervisor.tasks == set()


async def test_dispose_drains_within_the_grace() -> None:
    supervisor = TaskSupervisor(grace=timedelta(milliseconds=10))
    cancelled = asyncio.Event()

    async def work() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    supervisor.spawn(work())
    await supervisor.dispose()

    assert cancelled.is_set()


async def test_the_limit_caps_how_many_tasks_run_at_once() -> None:
    supervisor = TaskSupervisor(limit=2)
    release = asyncio.Event()
    running = 0
    peak = 0

    async def work() -> None:
        nonlocal peak, running
        running += 1
        peak = max(peak, running)
        await release.wait()
        running -= 1

    for _ in range(5):
        supervisor.spawn(work())

    await asyncio.sleep(0)
    release.set()

    assert await supervisor.drain(timedelta(seconds=1))
    assert peak == 2


async def test_spawn_returns_while_the_limit_is_exhausted() -> None:
    supervisor = TaskSupervisor(limit=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        started.set()
        await release.wait()

    async def queued() -> None:
        pass

    supervisor.spawn(hold())
    async with asyncio.timeout(1):
        await started.wait()

    # NOTE: the caller is a request handler; waiting for a slot happens in the task
    supervisor.spawn(queued())

    assert len(supervisor.tasks) == 2

    release.set()
    assert await supervisor.drain(timedelta(seconds=1))


async def test_a_task_cancelled_before_its_slot_closes_its_coroutine() -> None:
    supervisor = TaskSupervisor(limit=1)
    started = asyncio.Event()
    release = asyncio.Event()
    ran = False

    async def hold() -> None:
        started.set()
        await release.wait()

    async def queued() -> None:
        nonlocal ran
        ran = True

    coro = queued()
    supervisor.spawn(hold())
    async with asyncio.timeout(1):
        await started.wait()
    supervisor.spawn(coro)
    await asyncio.sleep(0)

    assert not await supervisor.drain(timedelta(milliseconds=10))
    assert not ran
    # NOTE: a coroutine that never ran must be closed, or the loop warns on GC
    assert getcoroutinestate(coro) == CORO_CLOSED


async def test_the_container_drains_the_supervisor_on_dispose() -> None:
    container = Container()
    container.bind(TaskSupervisorContract, instance=TaskSupervisor())
    done = asyncio.Event()

    async def work() -> None:
        await asyncio.sleep(0.01)
        done.set()

    async with container:
        container.resolve(TaskSupervisorContract).spawn(work())
        assert not done.is_set()

    assert done.is_set()


@pytest.mark.parametrize('limit', [0, -1])
def test_a_non_positive_limit_is_rejected(limit: int) -> None:
    with pytest.raises(ConcurrencyError, match='limit'):
        TaskSupervisor(limit=limit)


def test_a_negative_grace_is_rejected() -> None:
    with pytest.raises(ConcurrencyError, match='grace'):
        TaskSupervisor(grace=timedelta(seconds=-1))
