from collections.abc import Sequence
from typing import cast

from .lock import RedisLock
from .scripts import CLAIM, DRAIN


class RedisMailbox[T = bytes]:
    lock: RedisLock
    queue: str

    def __init__(self, lock: RedisLock, queue: str) -> None:
        self.lock = lock
        self.queue = queue

    async def claim(self, message: T) -> bool:
        async with self.lock.redis.connect(self.lock.alias) as client:
            script = client.register_script(CLAIM)
            claimed = await script(
                keys=[self.lock.key, self.queue],
                args=[self.lock.token, self.lock.lease(), message],
            )
            return bool(claimed)

    async def drain(self) -> tuple[T, ...]:
        # NOTE: an empty batch means the lock is gone
        async with self.lock.redis.connect(self.lock.alias) as client:
            script = client.register_script(DRAIN)
            backlog = await script(
                keys=[self.lock.key, self.queue],
                args=[self.lock.token],
            )
            return tuple(cast(Sequence[T], backlog))
