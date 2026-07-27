import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import override
from unittest.mock import AsyncMock, MagicMock

import httpx
import respx
from fastapi import FastAPI
from pydantic import SecretStr

from app.core.config.registry import HuginnConfig
from app.features.channels.config import ChannelConfigList
from app.features.channels.telegram.actor import TelegramChatActor
from app.features.channels.telegram.config import TelegramChannelConfig
from app.ports.channels import InboundMessage
from bin.server import build
from lib.mcp import MCPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract

_BASE_URL = 'https://api.telegram.org/botbot-token'
_PATH = '/webhooks/telegram'
_HEADER = 'X-Telegram-Bot-Api-Secret-Token'
_CHAT_ID = 4242


def update(update_id: int, *, chat_id: int = _CHAT_ID) -> dict[str, object]:
    # NOTE: a chat message is the only update shape the adapter forwards
    return {
        'update_id': update_id,
        'message': {
            'message_id': update_id,
            'date': 1_721_600_000,
            'chat': {'id': chat_id, 'type': 'private'},
            'text': 'hello',
        },
    }


class RecordingActor(TelegramChatActor):
    names: list[str]
    calls: list[str]
    started: asyncio.Event
    finished: asyncio.Event
    release: asyncio.Event
    block: bool
    delay: float

    def __init__(self, *, block: bool = False, delay: float = 0.0) -> None:
        self.names = []
        self.calls = []
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.release = asyncio.Event()
        self.block = block
        self.delay = delay

    @override
    async def handle(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
    ) -> None:
        self.names.append(message.channel)
        self.calls.append(message.event)
        self.started.set()
        if self.block:
            await self.release.wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        self.finished.set()


def telegram(secret: str) -> TelegramChannelConfig:
    return TelegramChannelConfig(
        kind='telegram',
        token=SecretStr('bot-token'),
        secret_token=SecretStr(secret),
    )


def make_config(channels: dict[str, TelegramChannelConfig]) -> HuginnConfig:
    return HuginnConfig(sections={'channels': ChannelConfigList(channels)})


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


@asynccontextmanager
async def webhook_client(
    config: HuginnConfig,
    redis: MagicMock,
    actor: TelegramChatActor,
) -> AsyncIterator[httpx.AsyncClient]:
    # NOTE: the ack reaches the real client; the gateway suite owns its payload
    respx.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )
    application = build()
    with (
        application.use(HuginnConfig, config),
        application.use(MCPClientContract, MagicMock(spec=MCPClientContract)),
        application.use(RedisContract, redis),
        application.use(TelegramChatActor, actor),
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


@respx.mock
async def test_rejects_missing_secret() -> None:
    handler = RecordingActor()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(set()), handler) as client:
        response = await client.post(_PATH, json=update(7))

    assert response.status_code == 403
    assert handler.calls == []


@respx.mock
async def test_rejects_unknown_secret() -> None:
    handler = RecordingActor()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(set()), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'wrong'},
            json=update(7),
        )

    assert response.status_code == 403
    assert handler.calls == []


@respx.mock
async def test_accepts_and_dispatches_to_the_handler() -> None:
    handler = RecordingActor()
    store = set()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(store), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=update(7),
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await handler.finished.wait()

    assert handler.calls == ['7']
    assert 'telegram:support_bot:7' in store


@respx.mock
async def test_routes_to_the_instance_matching_the_secret() -> None:
    handler = RecordingActor()
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
            json=update(9),
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await handler.finished.wait()

    assert handler.names == ['research_bot']
    assert handler.calls == ['9']


@respx.mock
async def test_deduplicates_repeated_update_id() -> None:
    handler = RecordingActor()
    store = set()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(store), handler) as client:
        first = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=update(7),
        )
        async with asyncio.timeout(1):
            await handler.finished.wait()
        handler.finished.clear()
        second = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=update(7),
        )
        await asyncio.sleep(0.05)

    assert first.status_code == 200
    assert second.status_code == 200
    assert handler.calls == ['7']


@respx.mock
async def test_returns_200_before_handler_completes() -> None:
    handler = RecordingActor(block=True)
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(set()), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=update(7),
        )
        assert response.status_code == 200

        async with asyncio.timeout(1):
            await handler.started.wait()
        assert not handler.finished.is_set()

        handler.release.set()
        async with asyncio.timeout(1):
            await handler.finished.wait()

    assert handler.calls == ['7']


@respx.mock
async def test_shutdown_drains_the_turn_the_webhook_already_acknowledged() -> None:
    # NOTE: the 200 is out, so uvicorn no longer counts this work as in-flight; the
    # only thing that waits for it is the supervisor's drain on the way out
    handler = RecordingActor(delay=0.05)
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(set()), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=update(7),
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await handler.started.wait()
        assert not handler.finished.is_set()

    assert handler.finished.is_set()


@respx.mock
async def test_processes_when_dedupe_is_unavailable() -> None:
    handler = RecordingActor()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(set(), fail=True), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json=update(7),
        )
        assert response.status_code == 200
        async with asyncio.timeout(1):
            await handler.finished.wait()

    assert handler.calls == ['7']


@respx.mock
async def test_acknowledges_but_discards_an_update_without_a_message() -> None:
    handler = RecordingActor()
    store = set()
    config = make_config({'support_bot': telegram('sekret')})

    async with webhook_client(config, make_redis(store), handler) as client:
        response = await client.post(
            _PATH,
            headers={_HEADER: 'sekret'},
            json={'update_id': 7, 'edited_message': {}},
        )

    # NOTE: the adapter normalises or discards; an update with no chat cannot be
    # keyed by conversation, so nothing downstream can act on it
    assert response.status_code == 200
    assert handler.calls == []
    assert not store
