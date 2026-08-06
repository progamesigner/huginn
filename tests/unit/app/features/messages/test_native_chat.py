import asyncio
import json
from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Any, cast, override
from unittest.mock import MagicMock

import httpx
from fastapi import FastAPI
from pydantic import SecretStr

from app.contracts.channels import ChannelReplyContract, TurnEnding
from app.core.config.registry import HuginnConfig
from app.features.agents.config import (
    AGENTS,
    DEFAULTS,
    AgentConfig,
    AgentConfigList,
    DefaultsConfig,
)
from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.types import TurnRequest
from app.features.auth.config import (
    API_KEYS,
    ApiKeyConfig,
    ApiKeyConfigList,
    ApiKeyRouterConfig,
    ApiKeyToolsConfig,
)
from bin.server import build
from lib.mcp import MCPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract
from tests.helpers.redis import FakeClient, make_redis

_PATH = '/v1/chat'
_CREDENTIAL = 'web-key'


class StreamingTurns(TurnHandlerContract):
    requests: list[TurnRequest]
    answer: tuple[str, ...]
    fail: bool
    ending: TurnEnding

    def __init__(
        self,
        *answer: str,
        fail: bool = False,
        ending: TurnEnding = 'complete',
    ) -> None:
        self.requests = []
        self.answer = answer or ('the ', 'answer')
        self.ending = ending
        self.fail = fail

    @override
    async def handle(
        self,
        request: TurnRequest,
        reply: ChannelReplyContract,
    ) -> None:
        self.requests.append(request)
        await reply.on_routed(request.policy.agent or 'concierge')
        for fragment in self.answer:
            await reply.on_text(fragment)

        if self.fail:
            raise RuntimeError('the model gave up')

        if self.ending != 'complete':
            await reply.unfinished(self.ending)

        await reply.finish()


class BusyClient(FakeClient):
    @override
    async def set(
        self,
        key: str,
        value: bytes,
        *,
        nx: bool = False,
        px: int | None = None,
    ) -> bool | None:
        # NOTE: what redis answers when the key is already there and nx was asked for
        return None


def api_key(
    *,
    agents: tuple[str, ...] = ('scout', 'concierge'),
    agent: str | None = None,
    router: ApiKeyRouterConfig | None = None,
    tools: ApiKeyToolsConfig | None = None,
) -> ApiKeyConfig:
    return ApiKeyConfig(
        token=SecretStr(_CREDENTIAL),
        agents=agents,
        agent=agent,
        router=router,
        tools=tools,
    )


def make_config(key: ApiKeyConfig) -> HuginnConfig:
    return HuginnConfig(
        sections={
            AGENTS.key: AgentConfigList(
                {
                    'scout': AgentConfig(description='research', instructions='scout'),
                    'concierge': AgentConfig(description='chat', instructions='hello'),
                },
            ),
            API_KEYS.key: ApiKeyConfigList({'web': key}),
            DEFAULTS.key: DefaultsConfig(model='gpt-5-mini'),
        },
    )


@asynccontextmanager
async def chat_client(
    config: HuginnConfig,
    turns: TurnHandlerContract,
    redis: MagicMock | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    application = build()
    with (
        application.use(HuginnConfig, config),
        application.use(MCPClientContract, MagicMock(spec=MCPClientContract)),
        application.use(RedisContract, redis or make_redis(FakeClient())),
        application.use(TurnHandlerContract, turns),
    ):
        app = application.resolve(RouterContract).app
        assert isinstance(app, FastAPI)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url='http://huginn',
            ) as client:
                yield client


@asynccontextmanager
async def asgi_app(
    config: HuginnConfig,
    turns: TurnHandlerContract,
) -> AsyncIterator[FastAPI]:
    application = build()
    with (
        application.use(HuginnConfig, config),
        application.use(MCPClientContract, MagicMock(spec=MCPClientContract)),
        application.use(RedisContract, make_redis(FakeClient())),
        application.use(TurnHandlerContract, turns),
    ):
        app = application.resolve(RouterContract).app
        assert isinstance(app, FastAPI)
        async with app.router.lifespan_context(app):
            yield app


def scope() -> dict[str, object]:
    return {
        'type': 'http',
        'asgi': {'version': '3.0'},
        'http_version': '1.1',
        'method': 'POST',
        'scheme': 'http',
        'path': _PATH,
        'raw_path': _PATH.encode(),
        'query_string': b'',
        'root_path': '',
        'headers': [
            (b'host', b'huginn'),
            (b'content-type', b'application/json'),
            (b'authorization', f'Bearer {_CREDENTIAL}'.encode()),
        ],
        'client': ('127.0.0.1', 12345),
        'server': ('huginn', 80),
    }


def events(body: str) -> list[tuple[str, str]]:
    return [
        (
            next(
                (
                    line.removeprefix('event: ')
                    for line in frame.splitlines()
                    if line.startswith('event: ')
                ),
                '',
            ),
            next(
                (
                    line.removeprefix('data: ')
                    for line in frame.splitlines()
                    if line.startswith('data: ')
                ),
                '',
            ),
        )
        for frame in body.split('\n\n')
        if frame
    ]


async def test_a_turn_streams_its_whole_life_as_events() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello'},
        )

    assert response.status_code == 200
    assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'
    assert [name for name, _ in events(response.text)] == [
        'opened',
        'routed',
        'delta',
        'delta',
        'done',
    ]


async def test_a_turn_that_stopped_short_says_so_before_it_is_done() -> None:
    turns = StreamingTurns(ending='timeout')

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello'},
        )

    named = events(response.text)
    assert [name for name, _ in named] == [
        'opened',
        'routed',
        'delta',
        'delta',
        'unfinished',
        'done',
    ]
    assert json.loads(dict(named)['unfinished']) == {'ending': 'timeout'}


async def test_native_request_forwards_max_completion_tokens() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello', 'max_completion_tokens': 4096},
        )

    assert response.status_code == 200
    assert turns.requests[0].max_completion_tokens == 4096


async def test_the_stream_is_not_buffered_on_the_way_out() -> None:
    # NOTE: an ingress that buffers turns a stream back into one late answer
    async with chat_client(make_config(api_key(agent='scout')), StreamingTurns()) as (
        client
    ):
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello'},
        )

    assert response.headers['cache-control'] == 'no-cache'
    assert response.headers['x-accel-buffering'] == 'no'


async def test_a_request_without_a_conversation_is_told_which_one_it_got() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello'},
        )

    opened = events(response.text)[0]
    assert opened[0] == 'opened'
    assert turns.requests[0].chat in opened[1]


async def test_a_request_may_continue_a_conversation_it_names() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello', 'conversation': 'c-1'},
        )

    assert turns.requests[0].chat == 'c-1'
    assert turns.requests[0].channel == 'web'


async def test_the_turn_is_attributed_to_the_key_that_opened_it() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello'},
        )

    prompt = turns.requests[0].prompts[0]
    assert (prompt.text, prompt.actor) == ('hello', 'client:web')


async def test_a_request_may_name_an_agent_the_key_allows() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key()), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello', 'agent': 'concierge'},
        )

    assert response.status_code == 200
    assert turns.requests[0].policy.agent == 'concierge'
    assert turns.requests[0].policy.allowed == frozenset({'scout', 'concierge'})


async def test_a_request_for_an_agent_the_key_may_not_use_is_forbidden() -> None:
    turns = StreamingTurns()

    async with chat_client(
        make_config(api_key(agents=('concierge',))),
        turns,
    ) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello', 'agent': 'scout'},
        )

    assert response.status_code == 403
    assert response.json()['reason'] == 'FORBIDDEN_AGENT'
    assert turns.requests == []


async def test_a_key_that_pins_no_agent_and_cannot_route_is_a_bad_request() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key()), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello'},
        )

    assert response.status_code == 400
    assert response.json()['reason'] == 'UNROUTABLE_REQUEST'
    assert turns.requests == []


async def test_a_key_with_a_router_leaves_the_agent_to_the_router() -> None:
    turns = StreamingTurns()
    key = api_key(router=ApiKeyRouterConfig(fallback='concierge'))

    async with chat_client(make_config(key), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello'},
        )

    assert response.status_code == 200
    policy = turns.requests[0].policy
    assert policy.agent is None
    assert policy.selection is not None
    assert policy.selection.fallback == 'concierge'


async def test_a_denied_tool_travels_with_the_turn() -> None:
    turns = StreamingTurns()
    key = api_key(agent='scout', tools=ApiKeyToolsConfig(deny=('dangerous-tool',)))

    async with chat_client(make_config(key), turns) as client:
        await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello'},
        )

    assert turns.requests[0].denied_tools == frozenset({'dangerous-tool'})


async def test_an_unauthenticated_request_never_reaches_a_turn() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        missing = await client.post(_PATH, json={'message': 'hello'})
        wrong = await client.post(
            _PATH,
            headers={'Authorization': 'Bearer nope'},
            json={'message': 'hello'},
        )

    assert (missing.status_code, wrong.status_code) == (401, 401)
    assert turns.requests == []


async def test_a_second_turn_on_a_busy_conversation_is_rejected() -> None:
    # NOTE: reject rather than queue -- a web client asks one question at a time, and a
    # 409 is something it can show
    turns = StreamingTurns()

    async with chat_client(
        make_config(api_key(agent='scout')),
        turns,
        make_redis(BusyClient()),
    ) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello', 'conversation': 'c-1'},
        )

    assert response.status_code == 409
    assert response.json()['reason'] == 'CONVERSATION_BUSY'
    assert turns.requests == []


async def test_a_failure_after_the_first_token_arrives_as_an_error_event() -> None:
    # NOTE: the status line is spent by then, so the only way to say it broke is an event
    turns = StreamingTurns('half an ', fail=True)

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello'},
        )

    assert response.status_code == 200
    assert [name for name, _ in events(response.text)] == [
        'opened',
        'routed',
        'delta',
        'error',
    ]


async def test_a_client_that_hangs_up_takes_the_turn_with_it() -> None:
    # NOTE: driven as raw ASGI on purpose -- a client that stops reading is only visible
    # as an `http.disconnect` message, which httpx's ASGI transport never sends, and the
    # whole point of this test is that the disconnect reaches the turn
    cancelled = asyncio.Event()
    delivered = asyncio.Event()

    class HangingTurns(TurnHandlerContract):
        @override
        async def handle(
            self,
            request: TurnRequest,
            reply: ChannelReplyContract,
        ) -> None:
            await reply.on_text('first')
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()

    async def receive() -> dict[str, object]:
        if not asked:
            asked.append(True)
            return {
                'type': 'http.request',
                'body': json.dumps({'message': 'hello'}).encode(),
                'more_body': False,
            }

        await delivered.wait()
        return {'type': 'http.disconnect'}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)
        if message['type'] == 'http.response.body' and b'first' in cast(
            bytes,
            message.get('body', b''),
        ):
            delivered.set()

    asked = []
    sent = []
    async with asgi_app(make_config(api_key(agent='scout')), HangingTurns()) as app:
        async with asyncio.timeout(2):
            await app(scope(), receive, send)
            await cancelled.wait()

    assert any(message['type'] == 'http.response.start' for message in sent)


async def test_a_request_with_no_message_is_rejected_before_a_turn() -> None:
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'conversation': 'c-1'},
        )

    assert response.status_code == 422
    assert turns.requests == []


async def test_an_unknown_field_in_the_request_is_ignored() -> None:
    # NOTE: our own contract is tolerant on the way in, so a client that sends a field
    # we do not read still gets an answer
    turns = StreamingTurns()

    async with chat_client(make_config(api_key(agent='scout')), turns) as client:
        response = await client.post(
            _PATH,
            headers={'Authorization': f'Bearer {_CREDENTIAL}'},
            json={'message': 'hello', 'temperature': 0.7},
        )

    assert response.status_code == 200
