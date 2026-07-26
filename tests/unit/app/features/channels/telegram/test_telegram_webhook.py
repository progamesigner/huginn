import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import override
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi import FastAPI
from pydantic import SecretStr

from app.core.config.registry import HuginnConfig
from app.exts.telegram.types import Update
from app.features.channels.config import ChannelsConfig
from app.features.channels.telegram.actor import TelegramChatActor
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.channels.telegram.handlers import TelegramUpdateHandler
from bin.server import build
from lib.mcp import MCPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract

_PATH = '/webhooks/telegram'
_HEADER = 'X-Telegram-Bot-Api-Secret-Token'


class RecordingHandler(TelegramUpdateHandler):
    names: list[str]
    calls: list[int]
    started: asyncio.Event
    finished: asyncio.Event
    release: asyncio.Event
    block: bool

    def __init__(self, *, block: bool = False) -> None:
        self.names = []
        self.calls = []
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.release = asyncio.Event()
        self.block = block

    @override
    async def handle(
        self,
        name: str,
        channel: TelegramChannelConfig,
        update: Update,
    ) -> None:
        self.names.append(name)
        self.calls.append(update.update_id)
        self.started.set()
        if self.block:
            await self.release.wait()
        self.finished.set()


def telegram(secret: str) -> TelegramChannelConfig:
    return TelegramChannelConfig(
        kind='telegram',
        token=SecretStr('bot-token'),
        secret_token=SecretStr(secret),
    )


def make_config(channels: dict[str, TelegramChannelConfig]) -> HuginnConfig:
    return HuginnConfig(sections={'channels': ChannelsConfig(channels)})


def make_redis(store: set[str], *, fail: bool = False) -> MagicMock:
    async def _set(
        key: str,
        value: object,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool | None:
        if key in store:
            return None
        store.add(key)
        return True

    client = MagicMock()
    client.set = _set
    connection = MagicMock()
    if fail:
        connection.__aenter__ = AsyncMock(side_effect=RuntimeError('redis unavailable'))
    else:
        connection.__aenter__ = AsyncMock(return_value=client)
    connection.__aexit__ = AsyncMock(return_value=False)

    redis = MagicMock(spec=RedisContract)
    redis.connect.return_value = connection
    return redis


def make_actor() -> MagicMock:
    # NOTE: this file exercises ack/dedupe only; the actor gets its own test suite
    actor = MagicMock(spec=TelegramChatActor)
    actor.handle = AsyncMock(return_value=None)
    return actor


@asynccontextmanager
async def webhook_client(
    config: HuginnConfig,
    redis: MagicMock,
    handler: TelegramUpdateHandler,
) -> AsyncIterator[httpx.AsyncClient]:
    application = build()
    with (
        application.use(HuginnConfig, config),
        application.use(MCPClientContract, MagicMock(spec=MCPClientContract)),
        application.use(RedisContract, redis),
        application.use(TelegramUpdateHandler, handler),
        application.use(TelegramChatActor, make_actor()),
    ):
        app = application.resolve(RouterContract).app
        assert isinstance(app, FastAPI)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url='http://huginn',
            ) as client:
                yield client


async def test_rejects_missing_secret() -> None:
    handler = RecordingHandler()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(set()), handler) as client:
        response = await client.post(_PATH, json={'update_id': 7})

    assert response.status_code == 403
    assert handler.calls == []


async def test_rejects_unknown_secret() -> None:
    handler = RecordingHandler()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(set()), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'wrong'},
            json={'update_id': 7},
        )

    assert response.status_code == 403
    assert handler.calls == []


async def test_accepts_and_dispatches_to_the_handler() -> None:
    handler = RecordingHandler()
    store: set[str] = set()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(store), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json={'update_id': 7},
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await handler.finished.wait()

    assert handler.calls == [7]
    assert 'telegram:support_bot:7' in store


async def test_routes_to_the_instance_matching_the_secret() -> None:
    handler = RecordingHandler()
    config = make_config(
        {
            'support_bot': telegram('support-secret'),
            'research_bot': telegram('research-secret'),
        },
    )

    async with webhook_client(config, make_redis(set()), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'research-secret'},
            json={'update_id': 9},
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await handler.finished.wait()

    assert handler.names == ['research_bot']
    assert handler.calls == [9]


async def test_deduplicates_repeated_update_id() -> None:
    handler = RecordingHandler()
    store: set[str] = set()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(store), handler) as client:
        first = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json={'update_id': 7},
        )
        async with asyncio.timeout(1):
            await handler.finished.wait()
        handler.finished.clear()
        second = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json={'update_id': 7},
        )
        await asyncio.sleep(0.05)

    assert first.status_code == 200
    assert second.status_code == 200
    assert handler.calls == [7]


async def test_returns_200_before_handler_completes() -> None:
    handler = RecordingHandler(block=True)
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(set()), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json={'update_id': 7},
        )
        assert response.status_code == 200

        async with asyncio.timeout(1):
            await handler.started.wait()
        assert not handler.finished.is_set()

        handler.release.set()
        async with asyncio.timeout(1):
            await handler.finished.wait()

    assert handler.calls == [7]


async def test_processes_when_dedupe_is_unavailable() -> None:
    handler = RecordingHandler()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(set(), fail=True), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json={'update_id': 7},
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await handler.finished.wait()

    assert handler.calls == [7]
