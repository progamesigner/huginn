from unittest.mock import AsyncMock, Mock

import pytest

from app.features.agents.config import AgentConfig
from app.features.agents.entities import (
    RoutingContext,
    RoutingPolicy,
    SelectionPolicy,
    TurnPrompt,
)
from app.features.agents.router import Router
from lib.decision import (
    ChoiceAnswer,
    DecisionContract,
    DecisionHTTPError,
    DecisionResponseError,
    DecisionResult,
    DecisionTimeoutError,
)
from tests.helpers.logging import make_logger

from .test_agent_router import FakeLLM


@pytest.mark.parametrize(
    ('choice', 'expected'),
    [('scout', 'scout'), ('__none__', 'fallback'), ('secret', 'fallback')],
)
async def test_decision_only_receives_authorized_candidates(
    choice: str,
    expected: str,
) -> None:
    decisions = Mock(spec=DecisionContract)
    client = Mock()
    client.decide = AsyncMock(
        return_value=DecisionResult(
            model='jev',
            answers={
                'agent': ChoiceAnswer(
                    choice=choice,
                    confidence=0.9,
                    probabilities={'scout': 0.9, 'fallback': 0.1},
                ),
            },
        ),
    )
    decisions.client_for.return_value = client
    agents = {
        name: AgentConfig(description=name) for name in ('scout', 'fallback', 'secret')
    }
    router = Router(agents, FakeLLM(), decisions=decisions, logger=make_logger())
    policy = RoutingPolicy(
        selection=SelectionPolicy(fallback='fallback', engine='decision', model='jev'),
        allowed=frozenset(('scout', 'fallback')),
    )
    routed = await router.resolve(
        policy,
        'chat',
        RoutingContext(current=(TurnPrompt(text='hello', actor='u'),)),
    )
    assert routed.name == expected
    request = client.decide.call_args.args[0]
    assert set(request.questions['agent'].criteria) == {'scout', 'fallback', '__none__'}
    assert request.state['current'] == ['hello']


@pytest.mark.parametrize(
    'error',
    [
        DecisionTimeoutError('private'),
        DecisionHTTPError(502, 'private'),
        DecisionResponseError('private'),
        RuntimeError('private'),
    ],
)
async def test_decision_failure_falls_back_without_logging_state(
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    decisions = Mock(spec=DecisionContract)
    decisions.client_for.return_value.decide = AsyncMock(side_effect=error)
    router = Router(
        {name: AgentConfig(description=name) for name in ('a', 'b')},
        FakeLLM(),
        decisions=decisions,
        logger=make_logger(),
    )
    selection = SelectionPolicy(fallback='a', engine='decision', model='jev')
    assert (
        await router.resolve(
            RoutingPolicy(selection=selection),
            'chat',
            'secret prompt',
        )
    ).name == 'a'
    assert 'private' not in caplog.text
    assert 'secret prompt' not in caplog.text


async def test_zero_one_and_missing_client() -> None:
    decisions = Mock(spec=DecisionContract)
    router = Router(
        {'a': AgentConfig(description='a'), 'b': AgentConfig(description='b')},
        FakeLLM(),
        decisions=decisions,
        logger=make_logger(),
    )
    selection = SelectionPolicy(fallback='a', engine='decision', model='jev')
    context = RoutingContext(current=(TurnPrompt(text='hi', actor='u'),))
    assert (
        await router.decide(selection, context, {})
    ).fallback_reason == 'no_candidates'
    assert (
        await router.decide(selection, context, {'a': router.agents['a']})
    ).choice == 'a'
    decisions.client_for.assert_not_called()
    decisions.client_for.return_value = None
    assert (
        await router.decide(selection, context, router.agents)
    ).fallback_reason == 'missing_client'
    assert selection.history_limit == 6
    assert selection.model_copy(update={'history_turns': 0}).history_limit == 0


def _answering(answer: ChoiceAnswer) -> Mock:
    decisions = Mock(spec=DecisionContract)
    decisions.client_for.return_value.decide = AsyncMock(
        return_value=DecisionResult(model='jev', answers={'agent': answer}),
    )
    return decisions


@pytest.mark.parametrize(
    ('min_confidence', 'min_margin', 'answer', 'reason'),
    [
        (None, None, ChoiceAnswer(choice='a', confidence=0.1), None),
        (0.5, None, ChoiceAnswer(choice='a', confidence=0.5), None),
        (0.5, None, ChoiceAnswer(choice='a', confidence=0.25), 'low_confidence'),
        (0.5, None, ChoiceAnswer(choice='a'), 'low_confidence'),
        (
            None,
            0.5,
            ChoiceAnswer(choice='a', probabilities={'a': 0.75, 'b': 0.25}),
            None,
        ),
        (
            None,
            0.75,
            ChoiceAnswer(choice='a', probabilities={'a': 0.75, 'b': 0.25}),
            'low_margin',
        ),
    ],
)
async def test_a_decision_below_its_threshold_falls_back(
    min_confidence: float | None,
    min_margin: float | None,
    answer: ChoiceAnswer,
    reason: str | None,
) -> None:
    agents = {name: AgentConfig(description=name) for name in ('a', 'b')}
    router = Router(
        agents,
        FakeLLM(),
        decisions=_answering(answer),
        logger=make_logger(),
    )
    selection = SelectionPolicy(
        fallback='b',
        engine='decision',
        model='jev',
        min_confidence=min_confidence,
        min_margin=min_margin,
    )

    decision = await router.decide(
        selection,
        RoutingContext(current=(TurnPrompt(text='hi', actor='u'),)),
        agents,
    )

    assert decision.fallback_reason == reason
