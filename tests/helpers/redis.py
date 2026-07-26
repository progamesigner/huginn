from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

from lib.redis import RedisContract


class FakeScript:
    client: FakeClient
    script: str

    def __init__(self, client: FakeClient, script: str) -> None:
        self.client = client
        self.script = script

    async def __call__(
        self,
        keys: Sequence[str] | None = None,
        args: Sequence[object] | None = None,
    ) -> object:
        self.client.evaluated.append(
            (self.script, tuple(keys or ()), tuple(args or ())),
        )
        return self.client.outcomes.pop(0) if self.client.outcomes else 1


class FakeClient:
    evaluated: list[tuple[str, tuple[str, ...], tuple[object, ...]]]
    outcomes: list[object]
    writes: list[tuple[str, bytes, bool, int | None]]

    def __init__(self, outcomes: Sequence[object] = ()) -> None:
        self.evaluated = []
        self.outcomes = list(outcomes)
        self.writes = []

    async def set(
        self,
        key: str,
        value: bytes,
        *,
        nx: bool = False,
        px: int | None = None,
    ) -> bool | None:
        self.writes.append((key, value, nx, px))
        return True

    def register_script(self, script: str) -> FakeScript:
        return FakeScript(self, script)


def make_redis(client: FakeClient) -> MagicMock:
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=client)
    connection.__aexit__ = AsyncMock(return_value=False)

    redis = MagicMock(spec=RedisContract)
    redis.connect.return_value = connection
    return redis
