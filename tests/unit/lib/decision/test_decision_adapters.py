import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, override

import httpx2
import pytest
from pydantic import JsonValue, SecretStr

from lib.decision import (
    Adapter,
    ChoiceQuestion,
    DecisionClient,
    DecisionHTTPError,
    DecisionManager,
    DecisionProviderConfig,
    DecisionRef,
    DecisionRequest,
    DecisionResponseError,
    DecisionTimeoutError,
    DecisionTransportError,
)
from lib.decision.adapters.typesafe import TypeSafeAdapter
from lib.http import HTTPClientContract
from tests.helpers.logging import make_logger

FIXTURES = Path(__file__).parent / 'fixtures'

type Handler = Callable[[httpx2.Request], httpx2.Response]


def _fixture(name: str) -> dict[str, JsonValue]:
    return json.loads((FIXTURES / name).read_text())


class MockHTTP(HTTPClientContract):
    handler: Handler
    requests: list[httpx2.Request]

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests = []

    @asynccontextmanager
    @override
    async def connect(
        self,
        base_url: httpx2.URL | str = '',
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[httpx2.AsyncClient]:
        def record(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return self.handler(request)

        async with httpx2.AsyncClient(
            base_url=base_url,
            headers=dict(headers or {}),
            transport=httpx2.MockTransport(record),
        ) as client:
            yield client


type Route = Literal['bifrost', 'typesafe', 'openrouter', 'openai']

# NOTE: OpenRouter speaks TypeSafe's own format, so it is the typesafe adapter pointed
# at another url; Bifrost's native and OpenAI-compatible decision endpoints use their
# respective adapters
ROUTES: dict[Route, tuple[Adapter, str | None, str, str]] = {
    'bifrost': (
        'bifrost',
        'http://bifrost:8080/v1/decisions',
        'bifrost_choice.json',
        'http://bifrost:8080/v1/decisions',
    ),
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
    'openai': (
        'openai',
        None,
        'openai_choice.json',
        'https://api.openai.com/v1/decisions',
    ),
}


def _adapter(
    route: Route,
    handler: Handler,
) -> tuple[DecisionClient, MockHTTP]:
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
    manager = DecisionManager((provider,), http, logger=make_logger())
    client = manager.client_for(DecisionRef(model='jev', provider=route))
    assert client is not None
    return client, http


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
    return lambda _: httpx2.Response(status, json=payload)


def _recorded(route: Route) -> JsonValue:
    _, _, fixture, _ = ROUTES[route]
    return _fixture(fixture)['response']


@pytest.mark.parametrize('route', list(ROUTES))
async def test_adapter_maps_a_recorded_choice(route: Route) -> None:
    client, http = _adapter(route, _answering(_recorded(route)))

    result = await client.decide(_request())

    recorded = _recorded(route)
    assert isinstance(recorded, dict)
    usage = recorded['usage']
    assert isinstance(usage, dict)
    expected = _recorded_answer(route)
    answer = result.answers['agent']
    assert answer.choice == 'assistant-advanced'
    assert answer.choice == expected.get('choice', expected.get('value'))
    assert answer.confidence == expected['confidence']
    assert answer.probabilities == _weights(expected['probabilities'])
    assert result.model == recorded['model']
    assert result.usage is not None
    assert result.usage.input_tokens == 507
    assert result.usage.output_tokens in {
        usage.get('output_tokens'),
        usage.get('completion_tokens'),
    }
    [sent] = http.requests
    assert str(sent.url) == ROUTES[route][3]
    assert sent.headers['Authorization'] == 'Bearer secret'


@pytest.mark.parametrize('route', ['bifrost', 'typesafe', 'openrouter'])
async def test_gateway_sends_the_question_in_its_wire_shape(route: Route) -> None:
    client, http = _adapter(route, _answering(_recorded(route)))

    await client.decide(_request())

    body = json.loads(http.requests[0].content)
    question = body['questions']['agent']
    assert question['kind' if route == 'bifrost' else 'type'] == 'choice'
    assert '__none__' in question['criteria']
    assert body['state'] == _request().state
    assert body['model'] == 'jev'


async def test_openai_sends_the_question_in_its_wire_shape() -> None:
    client, http = _adapter('openai', _answering(_recorded('openai')))

    await client.decide(_request())

    body = json.loads(http.requests[0].content)
    recorded = _fixture('openai_choice.json')['request']
    assert isinstance(recorded, dict)
    assert body['questions'] == recorded['questions']
    assert body['input'] == recorded['input']
    assert json.loads(body['input']) == _request().state
    assert body['model'] == 'jev'
    assert 'state' not in body


async def test_openai_sends_text_state_as_the_input_itself() -> None:
    client, http = _adapter('openai', _answering(_recorded('openai')))

    await client.decide(_request().model_copy(update={'state': 'hello'}))

    assert json.loads(http.requests[0].content)['input'] == 'hello'


async def test_openai_matches_an_unnamed_answer_by_its_position() -> None:
    payload = _with_answer('openai', name=None)
    client, _ = _adapter('openai', _answering(payload))

    result = await client.decide(_request())

    assert result.answers['agent'].choice == 'assistant-advanced'


async def test_openai_refuses_a_question_the_model_declined() -> None:
    payload = _recorded('openai')
    assert isinstance(payload, dict)
    client, _ = _adapter(
        'openai',
        _answering({**payload, 'answers': [{'type': 'refusal', 'name': 'agent'}]}),
    )

    with pytest.raises(DecisionResponseError, match="'agent' was refused"):
        await client.decide(_request())


async def test_openai_refuses_extra_answers() -> None:
    payload = _recorded('openai')
    assert isinstance(payload, dict)
    answers = payload['answers']
    assert isinstance(answers, list)
    assert len(answers) == 1
    client, _ = _adapter(
        'openai',
        _answering({**payload, 'answers': [*answers, answers[0]]}),
    )

    with pytest.raises(DecisionResponseError, match='unexpected number of answers'):
        await client.decide(_request())


async def test_openai_refuses_answers_named_for_the_wrong_question() -> None:
    payload = _recorded('openai')
    assert isinstance(payload, dict)
    answer = _recorded_answer('openai')
    request = _request()
    question = request.questions['agent']
    two_questions = DecisionRequest(
        model=request.model,
        state=request.state,
        questions={'agent': question, 'other': question},
    )
    client, _ = _adapter(
        'openai',
        _answering({**payload, 'answers': [{**answer, 'name': 'other'}, answer]}),
    )

    with pytest.raises(DecisionResponseError, match="as 'other'"):
        await client.decide(two_questions)


@pytest.mark.parametrize(
    ('route', 'upstream'),
    [
        ('bifrost', 'typesafe'),
        ('openai', None),
        ('openrouter', 'TypeSafe'),
        ('typesafe', None),
    ],
)
async def test_adapter_reports_the_upstream_a_gateway_names(
    route: Route,
    upstream: str | None,
) -> None:
    client, _ = _adapter(route, _answering(_recorded(route)))

    assert (await client.decide(_request())).upstream == upstream


def _recorded_answer(route: Route) -> dict[str, JsonValue]:
    payload = _recorded(route)
    assert isinstance(payload, dict)
    match payload['answers']:
        case [dict() as answer]:
            return answer
        case {'agent': dict() as answer}:
            return answer
        case _:
            raise AssertionError(f'{route} fixture has no answer for agent')


def _weights(probabilities: JsonValue) -> dict[str, JsonValue]:
    # NOTE: OpenAI weighs options as a list of {value, probability}; TypeSafe as a map
    match probabilities:
        case list():
            return {
                weighed['value']: weighed['probability']
                for weighed in probabilities
                if isinstance(weighed, dict) and isinstance(weighed['value'], str)
            }
        case dict():
            return probabilities
        case _:
            raise AssertionError('probabilities are neither a list nor a map')


def _weighing(route: Route, weights: dict[str, float]) -> JsonValue:
    if route == 'openai':
        return [
            {'value': value, 'probability': probability}
            for value, probability in weights.items()
        ]

    return dict(weights)


def _without_answers(route: Route) -> JsonValue:
    payload = _recorded(route)
    assert isinstance(payload, dict)
    return {**payload, 'answers': [] if route == 'openai' else {}}


def _with_answer(route: Route, **changes: JsonValue) -> JsonValue:
    payload = _recorded(route)
    assert isinstance(payload, dict)
    answer = {**_recorded_answer(route), **changes}
    return {**payload, 'answers': [answer] if route == 'openai' else {'agent': answer}}


def _broken(route: Route, broken: str) -> JsonValue:
    answer_field = 'value' if route == 'bifrost' else 'choice'
    kind_field = 'kind' if route == 'bifrost' else 'type'
    match broken:
        case 'missing':
            return _without_answers(route)
        case 'foreign':
            return _with_answer(route, **{answer_field: 'intruder'})
        case 'weighed':
            return _with_answer(
                route,
                probabilities=_weighing(route, {'intruder': 0.5}),
            )
        case 'overweight':
            return _with_answer(
                route,
                probabilities=_weighing(route, {'assistant-fast': 1.5}),
            )
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

    expected = (
        'unexpected number of answers'
        if route == 'openai' and broken == 'missing'
        else reason
    )
    with pytest.raises(DecisionResponseError, match=expected):
        await client.decide(_request())


@pytest.mark.parametrize('route', list(ROUTES))
async def test_adapter_refuses_a_body_that_is_not_json(route: Route) -> None:
    client, _ = _adapter(route, lambda _: httpx2.Response(200, text='<html>'))

    with pytest.raises(DecisionResponseError, match='not JSON'):
        await client.decide(_request())


@pytest.mark.parametrize(
    ('route', 'error', 'status', 'message'),
    [
        ('bifrost', 'missing_key', 401, 'virtual key is required'),
        ('bifrost', 'provider_blocked', 403, "Provider 'typesafe' is not allowed"),
        ('bifrost', 'native_provider_blocked', 403, "Provider 'typesafe' is not"),
        ('typesafe', 'bad_model', 400, 'Unknown model: jev-nope'),
        ('typesafe', 'bad_key', 401, 'Cannot authenticate'),
        ('typesafe', 'too_many', 400, 'Too many choices'),
        ('openai', 'bad_model', 404, 'The model `gpt-nope` does not exist'),
        ('openai', 'bad_key', 401, 'Incorrect API key provided'),
        ('openai', 'gateway_missing_key', 401, 'virtual key is required'),
        ('openai', 'gateway_provider_blocked', 403, "Provider 'typesafe' is not"),
        ('openrouter', 'bad_model', 400, 'typesafe/jev-nope does not exist'),
        ('openrouter', 'bad_key', 401, 'Missing Authentication header'),
        ('openrouter', 'too_many', 400, 'Too many choices'),
        ('openrouter', 'kind_field', 400, 'invalid_union'),
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
    client, http = _adapter(
        'typesafe',
        lambda _: httpx2.Response(429, text='slow down'),
    )

    with pytest.raises(DecisionHTTPError) as raised:
        await client.decide(_request())

    assert raised.value.status == 429
    assert len(http.requests) == 1


async def test_adapter_turns_a_timeout_into_a_decision_error() -> None:
    def hang(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout('too slow', request=request)

    client, http = _adapter('typesafe', hang)

    with pytest.raises(DecisionTimeoutError, match=r'2\.5s'):
        await client.decide(_request())

    assert http.requests[0].extensions['timeout']['read'] == 2.5


async def test_adapter_turns_a_connection_failure_into_a_decision_error() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('refused', request=request)

    client, _ = _adapter('typesafe', refuse)

    with pytest.raises(DecisionTransportError, match='ConnectError'):
        await client.decide(_request())


async def test_timeout_bounds_the_entire_gateway_call() -> None:
    class SlowHTTP(MockHTTP):
        @asynccontextmanager
        @override
        async def connect(
            self,
            base_url: httpx2.URL | str = '',
            headers: Mapping[str, str] | None = None,
        ) -> AsyncIterator[httpx2.AsyncClient]:
            await asyncio.Event().wait()
            async with super().connect(base_url, headers) as client:
                yield client

    provider = DecisionProviderConfig(
        name='slow',
        adapter='typesafe',
        api_key=SecretStr('test'),
        models=('jev',),
        timeout=0.01,
    )
    adapter = TypeSafeAdapter(provider, SlowHTTP(lambda request: httpx2.Response(200)))
    with pytest.raises(DecisionTimeoutError):
        await asyncio.wait_for(adapter.decide(_request()), timeout=1)
