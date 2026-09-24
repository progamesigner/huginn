import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, override

import httpx
import pytest
from pydantic import JsonValue, SecretStr

from lib.decision import (
    Adapter,
    ChoiceQuestion,
    DecisionHTTPError,
    DecisionProviderConfig,
    DecisionRequest,
    DecisionResponseError,
    DecisionTimeoutError,
    DecisionTransportError,
)
from lib.decision.adapters.base import HTTPDecisionAdapter
from lib.decision.adapters.bifrost import BifrostAdapter
from lib.decision.adapters.typesafe import TypeSafeAdapter
from lib.http import HTTPClientContract

FIXTURES = Path(__file__).parent / 'fixtures'

type Handler = Callable[[httpx.Request], httpx.Response]


def _fixture(name: str) -> dict[str, JsonValue]:
    return json.loads((FIXTURES / name).read_text())


class MockHTTP(HTTPClientContract):
    handler: Handler
    requests: list[httpx.Request]

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests = []

    @asynccontextmanager
    @override
    async def connect(
        self,
        base_url: httpx.URL | str = '',
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[httpx.AsyncClient]:
        def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self.handler(request)

        async with httpx.AsyncClient(
            base_url=base_url,
            headers=dict(headers or {}),
            transport=httpx.MockTransport(record),
        ) as client:
            yield client


type Route = Literal['typesafe', 'openrouter', 'bifrost']

# NOTE: OpenRouter speaks TypeSafe's own format, so it is the typesafe adapter pointed
# at another url; only Bifrost's decisions api needs an adapter of its own
ROUTES: dict[Route, tuple[Adapter, str | None, str, str]] = {
    'typesafe': (
        'typesafe',
        None,
        'typesafe_choice.json',
        'https://api.typesafe.ai/v1/systemone',
    ),
    'openrouter': (
        'typesafe',
        'https://openrouter.ai/api/alpha/decisions',
        'openrouter_choice.json',
        'https://openrouter.ai/api/alpha/decisions',
    ),
    'bifrost': (
        'bifrost',
        'http://bifrost:8080/v1/decisions',
        'bifrost_choice.json',
        'http://bifrost:8080/v1/decisions',
    ),
}
_ADAPTERS: dict[Adapter, type[HTTPDecisionAdapter]] = {
    'bifrost': BifrostAdapter,
    'typesafe': TypeSafeAdapter,
}


def _adapter(
    route: Route,
    handler: Handler,
) -> tuple[HTTPDecisionAdapter, MockHTTP]:
    http = MockHTTP(handler)
    adapter, url, _, _ = ROUTES[route]
    provider = DecisionProviderConfig(
        name=route,
        adapter=adapter,
        api_key=SecretStr('secret'),
        models=('jev',),
        url=url,
        timeout=2.5,
    )
    return _ADAPTERS[adapter](provider, http), http


def _request() -> DecisionRequest:
    recorded = _fixture('typesafe_choice.json')['request']
    assert isinstance(recorded, dict)
    questions = recorded['questions']
    assert isinstance(questions, dict)
    question = questions['agent']
    assert isinstance(question, dict)
    criteria = question['criteria']
    instructions = question['instructions']
    assert isinstance(criteria, dict)
    assert isinstance(instructions, str)
    return DecisionRequest(
        model='jev',
        state=recorded['state'] if isinstance(recorded['state'], dict) else {},
        questions={
            'agent': ChoiceQuestion(
                instructions=instructions,
                criteria={key: str(value) for key, value in criteria.items()},
            ),
        },
    )


def _answering(payload: JsonValue, status: int = 200) -> Handler:
    return lambda _: httpx.Response(status, json=payload)


def _recorded(route: Route) -> JsonValue:
    _, _, fixture, _ = ROUTES[route]
    return _fixture(fixture)['response']


@pytest.mark.parametrize('route', list(ROUTES))
async def test_adapter_maps_a_recorded_choice(route: Route) -> None:
    client, http = _adapter(route, _answering(_recorded(route)))

    result = await client.decide(_request())

    recorded = _recorded(route)
    assert isinstance(recorded, dict)
    answer = result.answers['agent']
    assert answer.choice == 'assistant-advanced'
    assert answer.confidence in {0.95, 0.96}
    assert answer.probabilities['assistant-advanced'] == pytest.approx(0.97)
    assert result.model == recorded['model']
    assert result.usage is not None
    assert (result.usage.input_tokens, result.usage.output_tokens) == (507, 57)
    [sent] = http.requests
    assert str(sent.url) == ROUTES[route][3]
    assert sent.headers['Authorization'] == 'Bearer secret'


@pytest.mark.parametrize('route', list(ROUTES))
async def test_adapter_sends_the_question_in_its_wire_shape(route: Route) -> None:
    client, http = _adapter(route, _answering(_recorded(route)))

    await client.decide(_request())

    body = json.loads(http.requests[0].content)
    question = body['questions']['agent']
    field = 'kind' if route == 'bifrost' else 'type'
    assert question[field] == 'choice'
    assert '__none__' in question['criteria']
    assert body['state'] == _request().state
    assert body['model'] == 'jev'


@pytest.mark.parametrize(
    ('route', 'upstream'),
    [('bifrost', 'typesafe'), ('openrouter', 'TypeSafe'), ('typesafe', None)],
)
async def test_adapter_reports_the_upstream_a_gateway_names(
    route: Route,
    upstream: str | None,
) -> None:
    client, _ = _adapter(route, _answering(_recorded(route)))

    assert (await client.decide(_request())).upstream == upstream


async def test_bifrost_refuses_an_answer_emulated_by_another_provider() -> None:
    payload = _recorded('bifrost')
    assert isinstance(payload, dict)
    client, _ = _adapter(
        'bifrost',
        _answering({**payload, 'extra_fields': {'provider': 'openai'}}),
    )

    with pytest.raises(DecisionResponseError, match='openai'):
        await client.decide(_request())


def _without_answers(route: Route) -> JsonValue:
    payload = _recorded(route)
    assert isinstance(payload, dict)
    return {**payload, 'answers': {}}


def _answer_field(route: Route) -> str:
    return 'value' if route == 'bifrost' else 'choice'


def _with_answer(route: Route, **changes: JsonValue) -> JsonValue:
    payload = _recorded(route)
    assert isinstance(payload, dict)
    answers = payload['answers']
    assert isinstance(answers, dict)
    answer = answers['agent']
    assert isinstance(answer, dict)
    return {**payload, 'answers': {'agent': {**answer, **changes}}}


def _broken(route: Route, broken: str) -> JsonValue:
    kind_field = 'kind' if route == 'bifrost' else 'type'
    match broken:
        case 'missing':
            return _without_answers(route)
        case 'foreign':
            return _with_answer(route, **{_answer_field(route): 'intruder'})
        case 'weighed':
            return _with_answer(route, probabilities={'intruder': 0.5})
        case 'overweight':
            return _with_answer(route, probabilities={'assistant-fast': 1.5})
        case 'kind':
            return _with_answer(route, **{kind_field: 'noul'})
        case _:
            return ['not', 'an', 'object']


@pytest.mark.parametrize('route', list(ROUTES))
@pytest.mark.parametrize(
    ('broken', 'reason'),
    [
        ('missing', 'not answered'),
        ('foreign', 'does not offer'),
        ('weighed', 'weighed options'),
        ('overweight', 'unexpected shape'),
        ('kind', 'unexpected shape'),
        ('shape', 'unexpected shape'),
    ],
)
async def test_adapter_refuses_an_answer_that_breaks_the_question(
    route: Route,
    broken: str,
    reason: str,
) -> None:
    client, _ = _adapter(route, _answering(_broken(route, broken)))

    with pytest.raises(DecisionResponseError, match=reason):
        await client.decide(_request())


@pytest.mark.parametrize('route', list(ROUTES))
async def test_adapter_refuses_a_body_that_is_not_json(route: Route) -> None:
    client, _ = _adapter(route, lambda _: httpx.Response(200, text='<html>'))

    with pytest.raises(DecisionResponseError, match='not JSON'):
        await client.decide(_request())


@pytest.mark.parametrize(
    ('route', 'error', 'status', 'message'),
    [
        ('typesafe', 'bad_model', 400, 'Unknown model: jev-nope'),
        ('typesafe', 'bad_key', 401, 'Cannot authenticate'),
        ('typesafe', 'too_many', 400, 'Too many choices'),
        ('bifrost', 'missing_key', 401, 'virtual key is required'),
        ('bifrost', 'provider_blocked', 403, "Provider 'typesafe' is not allowed"),
        ('bifrost', 'native_provider_blocked', 403, "Provider 'typesafe' is not"),
        ('openrouter', 'bad_model', 400, 'typesafe/jev-nope does not exist'),
        ('openrouter', 'bad_key', 401, 'Missing Authentication header'),
        ('openrouter', 'too_many', 400, 'Too many choices'),
        ('openrouter', 'kind_field', 400, 'invalid_union'),
        ('bifrost', 'upstream_auth', 401, 'Cannot authenticate'),
    ],
)
async def test_adapter_surfaces_recorded_errors(
    route: Route,
    error: str,
    status: int,
    message: str,
) -> None:
    errors = _fixture(f'{route}_errors.json')['errors']
    assert isinstance(errors, dict)
    recorded = errors[error]
    assert isinstance(recorded, dict)
    client, _ = _adapter(route, _answering(recorded['response'], status))

    with pytest.raises(DecisionHTTPError, match=message) as raised:
        await client.decide(_request())

    assert raised.value.status == status


async def test_adapter_surfaces_a_rate_limit_without_retrying() -> None:
    client, http = _adapter('typesafe', lambda _: httpx.Response(429, text='slow down'))

    with pytest.raises(DecisionHTTPError) as raised:
        await client.decide(_request())

    assert raised.value.status == 429
    assert len(http.requests) == 1


async def test_adapter_turns_a_timeout_into_a_decision_error() -> None:
    def hang(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout('too slow', request=request)

    client, http = _adapter('typesafe', hang)

    with pytest.raises(DecisionTimeoutError, match=r'2\.5s'):
        await client.decide(_request())

    assert http.requests[0].extensions['timeout']['read'] == 2.5


async def test_adapter_turns_a_connection_failure_into_a_decision_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused', request=request)

    client, _ = _adapter('typesafe', refuse)

    with pytest.raises(DecisionTransportError, match='ConnectError'):
        await client.decide(_request())
