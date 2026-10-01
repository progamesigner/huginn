import pytest
from pydantic import SecretStr, ValidationError

from lib.decision import (
    MAX_CHOICES,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionProviderConfig,
    DecisionProviderRegistry,
    DecisionRef,
    DecisionRequest,
)
from tests.helpers.logging import make_logger


def _provider(name: str, *models: str) -> DecisionProviderConfig:
    return DecisionProviderConfig(
        name=name,
        adapter='typesafe',
        api_key=SecretStr('secret'),
        models=models,
    )


def test_choice_question_needs_at_least_one_option() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions='Pick one', criteria={})


def test_choice_question_refuses_more_options_than_typesafe_accepts() -> None:
    criteria = {f'agent-{index}': 'x' for index in range(MAX_CHOICES + 1)}

    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions='Pick one', criteria=criteria)


def test_choice_question_accepts_the_maximum() -> None:
    criteria = {f'agent-{index}': 'x' for index in range(MAX_CHOICES)}

    assert len(ChoiceQuestion(instructions='Pick', criteria=criteria).criteria) == 255


def test_decision_request_carries_structured_state() -> None:
    request = DecisionRequest(
        model='jev-1.13.0',
        state={'current': ['hello'], 'previous_turns': []},
        questions={'agent': ChoiceQuestion(instructions='Pick', criteria={'a': 'x'})},
    )

    assert request.state == {'current': ['hello'], 'previous_turns': []}


def test_decision_request_needs_a_question() -> None:
    with pytest.raises(ValidationError):
        DecisionRequest(model='jev-1.13.0', state='hello', questions={})


@pytest.mark.parametrize('confidence', [-0.1, 1.1])
def test_choice_answer_refuses_confidence_outside_the_unit_range(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice='a', confidence=confidence)


def test_choice_answer_refuses_probabilities_outside_the_unit_range() -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice='a', probabilities={'a': 1.5})


def test_choice_answer_margin_is_the_gap_between_the_top_two() -> None:
    answer = ChoiceAnswer(choice='a', probabilities={'a': 0.7, 'b': 0.2, 'c': 0.1})

    assert answer.margin == pytest.approx(0.5)
    assert ChoiceAnswer(choice='a', probabilities={'a': 1.0}).margin == 1.0
    assert ChoiceAnswer(choice='a').margin is None


def test_gateway_provider_needs_a_url() -> None:
    with pytest.raises(ValidationError, match='needs a url'):
        DecisionProviderConfig(
            name='bifrost',
            adapter='bifrost',
            api_key=SecretStr('secret'),
            models=('typesafe/jev-1.13.0',),
        )


def test_typesafe_falls_back_to_its_public_endpoint() -> None:
    assert _provider('typesafe', 'jev-1.13.0').endpoint == (
        'https://api.typesafe.ai/v1/systemone'
    )


def test_provider_needs_a_model() -> None:
    with pytest.raises(ValidationError):
        _provider('typesafe')


def test_registry_resolves_a_bare_model_its_only_owner_offers() -> None:
    typesafe = _provider('typesafe', 'jev-1.13.0')
    registry = DecisionProviderRegistry((typesafe,), logger=make_logger())

    assert registry.resolve(DecisionRef(model='jev-1.13.0')) is typesafe
    assert registry.resolve(DecisionRef(model='jev-0')) is None


def test_registry_pins_a_model_to_the_named_provider() -> None:
    first = _provider('first', 'jev-1.13.0')
    second = _provider('second', 'jev-1.13.0')
    registry = DecisionProviderRegistry((first, second), logger=make_logger())

    assert (
        registry.resolve(DecisionRef(model='jev-1.13.0', provider='second')) is second
    )
    assert registry.resolve(DecisionRef(model='jev-1.13.0', provider='ghost')) is None
    assert registry.resolve(DecisionRef(model='jev-0', provider='first')) is None


def test_registry_refuses_a_bare_model_several_providers_claim() -> None:
    registry = DecisionProviderRegistry(
        (_provider('first', 'jev-1.13.0'), _provider('second', 'jev-1.13.0')),
        logger=make_logger(),
    )

    assert registry.resolve(DecisionRef(model='jev-1.13.0')) is None
