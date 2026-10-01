import asyncio
from collections.abc import Coroutine
from datetime import timedelta
from typing import Any, override

from structlog.stdlib import BoundLogger

from .contracts import TaskSupervisorContract
from .exceptions import ConcurrencyError

_GRACE = timedelta(seconds=30)


class TaskSupervisor(TaskSupervisorContract):
    grace: timedelta
    logger: BoundLogger
    slots: asyncio.Semaphore | None
    tasks: set[asyncio.Task[None]]

    def __init__(
        self,
        *,
        logger: BoundLogger,
        limit: int | None = None,
        grace: timedelta = _GRACE,
    ) -> None:
        if limit is not None and limit < 1:
            raise ConcurrencyError('task limit must be positive')

        if grace < timedelta(0):
            raise ConcurrencyError('drain grace must not be negative')

        self.grace = grace
        self.logger = logger
        self.slots = None if limit is None else asyncio.Semaphore(limit)
        self.tasks = set()

    @override
    async def dispose(self) -> None:
        await self.drain(self.grace)

    @override
    async def drain(self, grace: timedelta | None = None) -> bool:
        if not self.tasks:
            return True

        # NOTE: snapshot; a task reaping itself mutates the live set mid-wait
        seconds = None if grace is None else grace.total_seconds()
        _, pending = await asyncio.wait(set(self.tasks), timeout=seconds)
        if not pending:
            return True

        self.logger.warning(
            'Cancelling %d task(s) still running after the drain',
            len(pending),
        )
        for task in pending:
            task.cancel()

        await asyncio.gather(*pending, return_exceptions=True)
        return False

    @override
    def spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        # NOTE: hold a strong reference until done, else the loop GCs the task
        task = asyncio.create_task(self._run(coro))
        self.tasks.add(task)
        task.add_done_callback(self._reap)

    def _reap(self, task: asyncio.Task[None]) -> None:
        self.tasks.discard(task)

        if task.cancelled():
            return

        if (error := task.exception()) is not None:
            self.logger.error('Background task failed', exc_info=error)

    async def _run(self, coro: Coroutine[Any, Any, None]) -> None:
        if self.slots is None:
            await coro
            return

        try:
            await self.slots.acquire()
        except asyncio.CancelledError:
            # NOTE: the coroutine never ran, so close it or the loop warns on GC
            coro.close()
            raise

        try:
            await coro
        finally:
            self.slots.release()
