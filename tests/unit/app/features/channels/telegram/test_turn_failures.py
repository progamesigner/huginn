import json
from datetime import timedelta
from typing import override
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from pydantic import SecretStr
from structlog.contextvars import get_contextvars

from app.contracts.channels import InboundMessage
from app.core.config.registry import HuginnConfig
from app.exts.telegram.gateways import TelegramGatewayFactory
from app.features.channels.config import ChannelConfigList
from app.features.channels.telegram.actor import TelegramChatActor
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.channels.telegram.service import TelegramMessageHandler
from lib.concurrency import TaskSupervisor
from lib.redis import RedisContract
from tests.helpers.clock import FakeClock
from tests.helpers.http import MockHTTP, Route
from tests.helpers.logging import make_logger

_BASE_URL = 'https://api.telegram.org/botbot-token'
_CHANNEL = 'support_bot'
_GRACE = timedelta(seconds=1)


def turn_id() -> str | None:
    bound = get_contextvars().get('turn_id')
    return bound if isinstance(bound, str) else None


class FakeActor(TelegramChatActor):
    calls: list[str]
    turns: list[str | None]
    error: Exception | None

    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls = []
        self.turns = []
        self.error = error

    @override
    async def handle(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
    ) -> None:
        self.calls.append(message.event)
        self.turns.append(turn_id())
        if self.error is not None:
            raise self.error


def inbound(event: int = 7) -> InboundMessage:
    return InboundMessage(
        channel=_CHANNEL,
        chat='4242',
        event=str(event),
        id=str(event),
        kind='telegram',
        text='hello',
    )


def make_config() -> HuginnConfig:
    channel = TelegramChannelConfig(
        kind='telegram',
        token=SecretStr('bot-token'),
        secret_token=SecretStr('sekret'),
    )
    return HuginnConfig(sections={'channels': ChannelConfigList({_CHANNEL: channel})})


def make_redis() -> MagicMock:
    client = MagicMock()
    client.set = AsyncMock(return_value=True)
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(return_value=client)
    connection.__aexit__ = AsyncMock(return_value=False)

    redis = MagicMock(spec=RedisContract)
    redis.connect.return_value = connection
    return redis


def make_handler(
    http: MockHTTP,
    actor: TelegramChatActor,
    supervisor: TaskSupervisor,
) -> TelegramMessageHandler:
    return TelegramMessageHandler(
        config=make_config(),
        redis=make_redis(),
        gateways=TelegramGatewayFactory(
            http.manager(),
            FakeClock(),
            logger=make_logger(),
        ),
        clock=FakeClock(),
        actor=actor,
        supervisor=supervisor,
        logger=make_logger(),
    )


def reactions(route: Route) -> list[str]:
    return [
        json.loads(call.request.content)['reaction'][0]['emoji'] for call in route.calls
    ]


def mock_reactions(http: MockHTTP, status: int = 200) -> Route:
    return http.post(f'{_BASE_URL}/setMessageReaction').mock(
        return_value=httpx2.Response(
            status,
            json={'ok': status == 200, 'result': True},
        ),
    )


async def run_turn(
    handler: TelegramMessageHandler,
    supervisor: TaskSupervisor,
    *,
    event: int = 7,
) -> None:
    await handler.handle(inbound(event))

    assert await supervisor.drain(_GRACE)


async def test_a_completed_turn_keeps_only_the_reaction_it_started_with(
    http: MockHTTP,
) -> None:
    route = mock_reactions(http)
    actor = FakeActor()
    supervisor = TaskSupervisor(logger=make_logger())

    await run_turn(make_handler(http, actor, supervisor), supervisor)

    assert actor.calls == ['7']
    assert reactions(route) == ['\N{EYES}']


async def test_a_crashed_turn_replaces_the_ack_with_the_failure_reaction(
    http: MockHTTP,
) -> None:
    # NOTE: the runner is not wired yet, so every user-facing failure the plan names --
    # the LLM dying mid-stream, an edit rejected by telegram, every tool failing --
    # arrives here as an exception out of the turn, which is what this drives
    route = mock_reactions(http)
    actor = FakeActor(error=RuntimeError('llm stream died'))
    supervisor = TaskSupervisor(logger=make_logger())

    await run_turn(make_handler(http, actor, supervisor), supervisor)

    assert reactions(route) == ['\N{EYES}', '\N{SHOCKED FACE WITH EXPLODING HEAD}']


async def test_a_crashed_turn_is_reported_in_the_log(
    http: MockHTTP,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_reactions(http)
    actor = FakeActor(error=RuntimeError('llm stream died'))
    supervisor = TaskSupervisor(logger=make_logger())
    handler = make_handler(http, actor, supervisor)

    with caplog.at_level('ERROR'):
        await run_turn(handler, supervisor)

    assert any('Turn failed' in record.message for record in caplog.records)


async def test_a_failed_acknowledgement_does_not_cost_the_user_their_turn(
    http: MockHTTP,
) -> None:
    # NOTE: acknowledge runs before the turn, so before this an unreachable telegram at
    # that moment took the whole answer down with it
    route = mock_reactions(http, status=400)
    actor = FakeActor()
    supervisor = TaskSupervisor(logger=make_logger())

    await run_turn(make_handler(http, actor, supervisor), supervisor)

    assert route.called
    assert actor.calls == ['7']


async def test_a_failed_report_leaves_nothing_for_the_supervisor_to_catch(
    http: MockHTTP,
    caplog: pytest.LogCaptureFixture,
) -> None:
    route = mock_reactions(http, status=400)
    actor = FakeActor(error=RuntimeError('llm stream died'))
    supervisor = TaskSupervisor(logger=make_logger())
    handler = make_handler(http, actor, supervisor)

    with caplog.at_level('ERROR'):
        await run_turn(handler, supervisor)

    # NOTE: both the ack and the report were refused, so the turn had nothing left to
    # tell the user; it still must not resurface as an unhandled task failure
    assert route.call_count == 2
    assert not [
        record
        for record in caplog.records
        if record.name == 'lib.concurrency.supervisor'
    ]


async def test_each_turn_runs_under_its_own_turn_id(http: MockHTTP) -> None:
    mock_reactions(http)
    actor = FakeActor()
    supervisor = TaskSupervisor(logger=make_logger())
    handler = make_handler(http, actor, supervisor)

    await run_turn(handler, supervisor, event=7)
    await run_turn(handler, supervisor, event=8)

    assert all(bound is not None for bound in actor.turns)
    assert len(set(actor.turns)) == 2
