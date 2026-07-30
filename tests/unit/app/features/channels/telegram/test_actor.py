from collections.abc import Awaitable, Callable
from datetime import timedelta

from pydantic import SecretStr

from app.features.channels.telegram.actor import TelegramChatActor
from app.features.channels.telegram.config import TelegramChannelConfig
from app.ports.channels import InboundMessage
from tests.helpers.redis import FakeClient, make_redis

_TTL = timedelta(seconds=60)

type Rounds = list[tuple[str, ...]]
type Processor = Callable[
    [TelegramChannelConfig, tuple[InboundMessage, ...]],
    Awaitable[None],
]


def channel() -> TelegramChannelConfig:
    return TelegramChannelConfig(
        kind='telegram',
        token=SecretStr('bot-token'),
        secret_token=SecretStr('sekret'),
    )


def inbound(event: int, chat: int = 555) -> InboundMessage:
    return InboundMessage(
        channel='support_bot',
        chat=str(chat),
        event=str(event),
        id=str(event),
        text=f'message {event}',
    )


def queued(event: int, chat: int = 555) -> bytes:
    return inbound(event, chat).model_dump_json().encode()


def collector() -> tuple[Rounds, Processor]:
    rounds = []

    async def process(
        ch: TelegramChannelConfig,
        messages: tuple[InboundMessage, ...],
    ) -> None:
        rounds.append(tuple(message.event for message in messages))

    return rounds, process


async def test_a_refused_claim_leaves_processing_to_the_owner() -> None:
    rounds, process = collector()
    actor = TelegramChatActor(make_redis(FakeClient(outcomes=[0])), process, ttl=_TTL)

    await actor.handle(channel(), inbound(1))

    assert rounds == []


async def test_the_owner_processes_its_own_message_then_stops_on_an_empty_drain() -> (
    None
):
    rounds, process = collector()
    client = FakeClient(outcomes=[1, []])
    actor = TelegramChatActor(make_redis(client), process, ttl=_TTL)

    await actor.handle(channel(), inbound(1))

    assert rounds == [('1',)]


async def test_a_non_empty_drain_coalesces_into_a_second_round() -> None:
    rounds, process = collector()
    client = FakeClient(outcomes=[1, [queued(2), queued(3)], []])
    actor = TelegramChatActor(make_redis(client), process, ttl=_TTL)

    await actor.handle(channel(), inbound(1))

    assert rounds == [('1',), ('2', '3')]


async def test_unreadable_queued_bytes_are_dropped_from_the_round() -> None:
    rounds, process = collector()
    client = FakeClient(outcomes=[1, [b'not json', queued(2)], []])
    actor = TelegramChatActor(make_redis(client), process, ttl=_TTL)

    await actor.handle(channel(), inbound(1))

    assert rounds == [('1',), ('2',)]


async def test_the_claim_queues_the_port_message_itself() -> None:
    _, process = collector()
    client = FakeClient(outcomes=[1, []])
    actor = TelegramChatActor(make_redis(client), process, ttl=_TTL)
    message = inbound(1)

    await actor.handle(channel(), message)

    # NOTE: the queue has to round-trip the port, not the vendor payload it came from
    _, keys, args = client.evaluated[0]
    payload = args[-1]

    assert keys == ('telegram:support_bot:lock:555', 'telegram:support_bot:mailbox:555')
    assert isinstance(payload, bytes)
    assert InboundMessage.model_validate_json(payload) == message
