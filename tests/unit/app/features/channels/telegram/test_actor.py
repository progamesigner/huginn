from datetime import timedelta

from pydantic import SecretStr

from app.exts.telegram.types import Chat, Message, Update
from app.features.channels.telegram.actor import TelegramChatActor
from app.features.channels.telegram.config import TelegramChannelConfig
from tests.helpers.redis import FakeClient, make_redis

_TTL = timedelta(seconds=60)


def channel() -> TelegramChannelConfig:
    return TelegramChannelConfig(
        kind='telegram',
        token=SecretStr('bot-token'),
        secret_token=SecretStr('sekret'),
    )


def body(update_id: int, chat_id: int = 555) -> bytes:
    update = Update(
        update_id=update_id,
        message=Message(
            message_id=update_id,
            date=0,
            chat=Chat(id=chat_id, type='private'),
        ),
    )
    return update.model_dump_json().encode()


async def test_updates_without_a_message_skip_the_actor_entirely() -> None:
    calls = []

    async def process(
        name: str,
        ch: TelegramChannelConfig,
        updates: tuple[Update, ...],
    ) -> None:
        calls.append(updates)

    client = FakeClient()
    actor = TelegramChatActor(make_redis(client), process, ttl=_TTL)

    await actor.handle(
        'support_bot',
        channel(),
        Update(update_id=1),
        b'{"update_id": 1}',
    )

    assert calls == []
    assert client.evaluated == []


async def test_a_refused_claim_leaves_processing_to_the_owner() -> None:
    calls = []

    async def process(
        name: str,
        ch: TelegramChannelConfig,
        updates: tuple[Update, ...],
    ) -> None:
        calls.append(updates)

    client = FakeClient(outcomes=[0])
    actor = TelegramChatActor(make_redis(client), process, ttl=_TTL)

    raw = body(1)
    await actor.handle('support_bot', channel(), Update.model_validate_json(raw), raw)

    assert calls == []


async def test_the_owner_processes_its_own_message_then_stops_on_an_empty_drain() -> (
    None
):
    calls = []

    async def process(
        name: str,
        ch: TelegramChannelConfig,
        updates: tuple[Update, ...],
    ) -> None:
        calls.append(tuple(update.update_id for update in updates))

    client = FakeClient(outcomes=[1, []])
    actor = TelegramChatActor(make_redis(client), process, ttl=_TTL)

    raw = body(1)
    await actor.handle('support_bot', channel(), Update.model_validate_json(raw), raw)

    assert calls == [(1,)]


async def test_a_non_empty_drain_coalesces_into_a_second_round() -> None:
    calls = []

    async def process(
        name: str,
        ch: TelegramChannelConfig,
        updates: tuple[Update, ...],
    ) -> None:
        calls.append(tuple(update.update_id for update in updates))

    client = FakeClient(outcomes=[1, [body(2), body(3)], []])
    actor = TelegramChatActor(make_redis(client), process, ttl=_TTL)

    raw = body(1)
    await actor.handle('support_bot', channel(), Update.model_validate_json(raw), raw)

    assert calls == [(1,), (2, 3)]


async def test_unparseable_queued_bytes_are_dropped_from_the_round() -> None:
    calls = []

    async def process(
        name: str,
        ch: TelegramChannelConfig,
        updates: tuple[Update, ...],
    ) -> None:
        calls.append(tuple(update.update_id for update in updates))

    client = FakeClient(outcomes=[1, [b'not json', body(2)], []])
    actor = TelegramChatActor(make_redis(client), process, ttl=_TTL)

    raw = body(1)
    await actor.handle('support_bot', channel(), Update.model_validate_json(raw), raw)

    assert calls == [(1,), (2,)]
