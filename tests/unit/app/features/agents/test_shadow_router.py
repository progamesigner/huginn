import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app.features.agents.config import AgentConfig
from app.features.agents.entities import (
    RoutingContext,
    RoutingPolicy,
    SelectionPolicy,
    ShadowPolicy,
    TurnPrompt,
)
from app.features.agents.router import Router
from app.features.agents.shadow_report import shadow_records
from lib.concurrency import TaskSupervisor
from lib.decision import (
    ChoiceAnswer,
    DecisionContract,
    DecisionResult,
    DecisionTimeoutError,
)
from lib.llm import TextChunk

from .test_agent_router import FakeClient, FakeLLM

_CONTEXT = RoutingContext(current=(TurnPrompt(text='hello', actor='u'),))
_SHADOWED = RoutingPolicy(
    selection=SelectionPolicy(
        fallback='b',
        model='chat',
        shadow=ShadowPolicy(model='jev', provider='gateway'),
    ),
)


def _agents() -> dict[str, AgentConfig]:
    return {name: AgentConfig(description=name) for name in ('a', 'b')}


def _llm_choosing(name: str) -> FakeLLM:
    return FakeLLM(FakeClient([TextChunk(text=name)]))


def _jev(decide: AsyncMock) -> Mock:
    decisions = Mock(spec=DecisionContract)
    decisions.client_for.return_value.decide = decide
    return decisions


def _answer(choice: str) -> DecisionResult:
    return DecisionResult(
        model='jev',
        answers={'agent': ChoiceAnswer(choice=choice, confidence=0.75)},
    )


async def test_the_shadow_never_holds_up_the_live_route(
    caplog: pytest.LogCaptureFixture,
) -> None:
    released = asyncio.Event()

    async def slow(_request: object) -> DecisionResult:
        await released.wait()
        return _answer('a')

    shadows = TaskSupervisor()
    router = Router(
        _agents(),
        _llm_choosing('a'),
        decisions=_jev(AsyncMock(side_effect=slow)),
        shadows=shadows,
    )

    routed = await router.resolve(_SHADOWED, 'chat-1', _CONTEXT)

    assert routed.name == 'a'
    assert shadows.tasks
    released.set()
    with caplog.at_level('INFO'):
        await router.dispose()
    assert 'Router shadow engine=decision choice=a primary=a agree=True' in caplog.text


async def test_a_disagreeing_shadow_is_logged_against_the_live_choice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = Router(
        _agents(),
        _llm_choosing('a'),
        decisions=_jev(AsyncMock(return_value=_answer('b'))),
        shadows=TaskSupervisor(),
    )

    with caplog.at_level('INFO'):
        routed = await router.resolve(_SHADOWED, 'chat-1', _CONTEXT)
        await router.dispose()

    assert routed.name == 'a'
    assert 'choice=b primary=a agree=False' in caplog.text
    assert [
        (record.primary, record.choice, record.confidence, record.reason)
        for record in shadow_records(caplog.messages)
    ] == [('a', 'b', 0.75, None)]


async def test_a_failing_shadow_leaves_the_live_route_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = Router(
        _agents(),
        _llm_choosing('a'),
        decisions=_jev(AsyncMock(side_effect=DecisionTimeoutError('private'))),
        shadows=TaskSupervisor(),
    )

    with caplog.at_level('INFO'):
        routed = await router.resolve(_SHADOWED, 'chat-1', _CONTEXT)
        await router.dispose()

    assert routed.name == 'a'
    assert 'reason=DecisionTimeoutError' in caplog.text
    assert 'private' not in caplog.text


async def test_a_router_without_a_shadow_never_asks_the_decision_engine() -> None:
    decide = AsyncMock(return_value=_answer('b'))
    shadows = Mock(spec=TaskSupervisor)
    router = Router(
        _agents(),
        _llm_choosing('a'),
        decisions=_jev(decide),
        shadows=shadows,
    )
    policy = RoutingPolicy(selection=SelectionPolicy(fallback='b', model='chat'))

    assert (await router.resolve(policy, 'chat-1', _CONTEXT)).name == 'a'

    shadows.spawn.assert_not_called()
    decide.assert_not_called()
