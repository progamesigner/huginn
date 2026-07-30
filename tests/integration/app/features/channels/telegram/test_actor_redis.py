import asyncio
import time
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from pydantic import SecretStr

from app.features.channels.telegram.actor import TelegramChatActor
from app.features.channels.telegram.config import TelegramChannelConfig
from app.ports.channels import InboundMessage
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


def message(event: int, chat: int) -> InboundMessage:
    return InboundMessage(
        channel='support_bot',
        chat=str(chat),
        event=str(event),
        id=str(event),
        kind='telegram',
        text=f'message {event}',
    )


class SettlingProcessor:
    rounds: list[tuple[str, ...]]
    settle: float
    calls: int

    def __init__(self, *, settle: float) -> None:
        self.rounds = []
        self.settle = settle
        self.calls = 0

    async def __call__(
        self,
        channel: TelegramChannelConfig,
        messages: tuple[InboundMessage, ...],
    ) -> None:
        self.calls += 1
        if self.calls == 1:
            # NOTE: hold the first round open so the other concurrent claims queue
            await asyncio.sleep(self.settle)
        self.rounds.append(tuple(message.event for message in messages))


async def test_concurrent_messages_to_one_chat_all_process_within_two_rounds(
    redis: RedisManager,
) -> None:
    processor = SettlingProcessor(settle=0.2)
    actor = TelegramChatActor(redis, processor, ttl=_TTL)
    chat_id = 12345

    messages = [message(i, chat_id) for i in range(1, 6)]
    await asyncio.gather(*(actor.handle(_CHANNEL, item) for item in messages))

    assert len(processor.rounds) <= 2
    processed = sorted(event for round_ in processor.rounds for event in round_)
    assert processed == ['1', '2', '3', '4', '5']


async def test_two_chats_process_concurrently_without_blocking_each_other(
    redis: RedisManager,
) -> None:
    a = SettlingProcessor(settle=0.2)
    b = SettlingProcessor(settle=0.2)
    actor_a = TelegramChatActor(redis, a, ttl=_TTL)
    actor_b = TelegramChatActor(redis, b, ttl=_TTL)

    started = time.perf_counter()
    await asyncio.gather(
        actor_a.handle(_CHANNEL, message(1, chat=111)),
        actor_b.handle(_CHANNEL, message(2, chat=222)),
    )
    elapsed = time.perf_counter() - started

    # NOTE: serialized processing would take ~2x settle; concurrent stays near 1x
    assert elapsed < 0.35
    assert a.rounds == [('1',)]
    assert b.rounds == [('2',)]
