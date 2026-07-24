import asyncio
import logging

import pytest

from app.features.channels.telegram.tasks import BackgroundTasks


async def test_spawn_runs_and_reaps_the_task() -> None:
    tasks = BackgroundTasks()
    done = asyncio.Event()

    async def work() -> None:
        done.set()

    tasks.spawn(work())
    assert len(tasks.tasks) == 1

    async with asyncio.timeout(1):
        await done.wait()
    await asyncio.sleep(0)

    assert tasks.tasks == set()


async def test_background_exception_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    tasks = BackgroundTasks()

    async def boom() -> None:
        raise RuntimeError('kaboom')

    tasks.spawn(boom())
    task = next(iter(tasks.tasks))
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert 'Background task failed' in caplog.text
    assert tasks.tasks == set()
