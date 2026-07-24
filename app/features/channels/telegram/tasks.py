import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundTasks:
    tasks: set[asyncio.Task[None]]

    def __init__(self) -> None:
        self.tasks = set()

    def spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        # NOTE: hold a strong reference until done, else the loop GCs the task
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self._reap)

    def _reap(self, task: asyncio.Task[None]) -> None:
        self.tasks.discard(task)

        if task.cancelled():
            return

        if (error := task.exception()) is not None:
            logger.error('Background task failed', exc_info=error)
