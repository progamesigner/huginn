import asyncio
import time
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from pydantic import SecretStr

from app.exts.telegram.types import Chat, Message, Update
from app.features.channels.telegram.actor import TelegramChatActor
from app.features.channels.telegram.config import TelegramChannelConfig
from lib.redis import RedisManager

pytestmark = pytest.mark.integration

_CHANNEL = TelegramChannelConfig(
    kind='telegram',
    token=SecretStr('bot-token'),
    secret_token=SecretStr('sekret'),
)
_TTL = timedelta(seconds=5)


@pytest.fixture
async def redis(redis_url: str) -> AsyncIterator[RedisManager]:
    manager = RedisManager({None: redis_url})
    try:
        yield manager
    finally:
        await manager.dispose()


def message(update_id: int, chat_id: int) -> tuple[Update, bytes]:
    update = Update(
        update_id=update_id,
        message=Message(
            message_id=update_id,
            date=0,
            chat=Chat(id=chat_id, type='private'),
        ),
    )
    return update, update.model_dump_json().encode()


class SettlingProcessor:
    def __init__(self, *, settle: float) -> None:
        self.rounds: list[tuple[int, ...]] = []
        self.settle = settle
        self.calls = 0

    async def __call__(
        self,
        name: str,
        channel: TelegramChannelConfig,
        updates: tuple[Update, ...],
    ) -> None:
        self.calls += 1
        if self.calls == 1:
            # NOTE: hold the first round open so the other concurrent claims queue
            await asyncio.sleep(self.settle)
        self.rounds.append(tuple(update.update_id for update in updates))


async def test_concurrent_messages_to_one_chat_all_process_within_two_rounds(
    redis: RedisManager,
) -> None:
    processor = SettlingProcessor(settle=0.2)
    actor = TelegramChatActor(redis, processor, ttl=_TTL)
    chat_id = 12345

    pairs = [message(i, chat_id) for i in range(1, 6)]
    await asyncio.gather(
        *(actor.handle('support_bot', _CHANNEL, update, body) for update, body in pairs)
    )

    assert len(processor.rounds) <= 2
    processed = sorted(i for round_ in processor.rounds for i in round_)
    assert processed == [1, 2, 3, 4, 5]


async def test_two_chats_process_concurrently_without_blocking_each_other(
    redis: RedisManager,
) -> None:
    a = SettlingProcessor(settle=0.2)
    b = SettlingProcessor(settle=0.2)
    actor_a = TelegramChatActor(redis, a, ttl=_TTL)
    actor_b = TelegramChatActor(redis, b, ttl=_TTL)

    update_a, body_a = message(1, chat_id=111)
    update_b, body_b = message(2, chat_id=222)

    started = time.perf_counter()
    await asyncio.gather(
        actor_a.handle('support_bot', _CHANNEL, update_a, body_a),
        actor_b.handle('support_bot', _CHANNEL, update_b, body_b),
    )
    elapsed = time.perf_counter() - started

    # NOTE: serialized processing would take ~2x settle; concurrent stays near 1x
    assert elapsed < 0.35
    assert a.rounds == [(1,)]
    assert b.rounds == [(2,)]
