from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import httpx2
import pytest
from fastapi import FastAPI

from app.core.config.registry import HuginnConfig
from app.features.agents.contracts import TurnServiceContract
from app.features.blobs.contracts import BlobServiceContract
from app.features.identities.contracts import (
    IdentityLinkingServiceContract,
    IdentityManagementServiceContract,
    IdentityResolutionServiceContract,
)
from app.features.identities.entities import AnonymousPrincipal, LinkedPrincipal
from app.features.identities.exceptions import InvalidIdentityLinkError
from bin.server import build
from lib.mcp import MCPClientContract
from lib.redis import RedisContract
from lib.result import Err, Ok
from lib.router import RouterContract
from tests.helpers.blobs import MemoryBlobs
from tests.helpers.redis import FakeClient, make_redis

from .test_native_chat import StreamingTurns, api_key, make_config

_PRINCIPAL = LinkedPrincipal(
    user_id=UUID(int=31),
    external_identity_id=UUID(int=32),
    handle='fictional',
    memory_scope='prior-memory',
    display_name='Example Person',
)


def headers(user: str | None = 'app-user-1') -> dict[str, str]:
    result = {'Authorization': 'Bearer web-key'}
    if user is not None:
        result['X-Huginn-User'] = user
    return result


@asynccontextmanager
async def client_for(
    turns: StreamingTurns,
    resolver: MagicMock,
    management: MagicMock | None = None,
) -> AsyncIterator[httpx2.AsyncClient]:
    application = build()
    with (
        application.use(BlobServiceContract, MemoryBlobs()),
        application.use(HuginnConfig, make_config(api_key(agent='scout'))),
        application.use(MCPClientContract, MagicMock(spec=MCPClientContract)),
        application.use(RedisContract, make_redis(FakeClient())),
        application.use(TurnServiceContract, turns),
        application.use(IdentityResolutionServiceContract, resolver),
        application.use(
            IdentityManagementServiceContract,
            management or MagicMock(spec=IdentityManagementServiceContract),
        ),
        application.use(
            IdentityLinkingServiceContract,
            management or MagicMock(spec=IdentityLinkingServiceContract),
        ),
    ):
        app = application.resolve(RouterContract).app
        assert isinstance(app, FastAPI)
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app),
                base_url='http://huginn',
            ) as client,
        ):
            yield client


@pytest.mark.parametrize(
    'path',
    ['/v1/chat', '/openai/v1/chat/completions', '/anthropic/v1/messages'],
)
async def test_all_chat_surfaces_use_asserted_identity_and_isolate_threads(
    path: str,
) -> None:
    turns = StreamingTurns()
    resolver = MagicMock(spec=IdentityResolutionServiceContract)
    resolver.resolve = AsyncMock(return_value=_PRINCIPAL)
    ask = (
        {'message': 'hello', 'conversation': 'shared-thread'}
        if path == '/v1/chat'
        else {
            'messages': [{'role': 'user', 'content': 'hello'}],
            'user': 'shared-thread',
        }
    )
    async with client_for(turns, resolver) as client:
        first = await client.post(path, headers=headers(), json=ask)
        second = await client.post(path, headers=headers('app-user-2'), json=ask)
        machine = await client.post(path, headers=headers(None), json=ask)
    assert [response.status_code for response in (first, second, machine)] == [
        200,
        200,
        200,
    ]
    assert resolver.resolve.await_args_list[0].args == ('http', 'web', 'app-user-1')
    assert resolver.resolve.await_args_list[1].args == ('http', 'web', 'app-user-2')
    assert resolver.resolve.await_count == 2
    assert turns.requests[0].channel != turns.requests[1].channel
    assert turns.requests[0].chat == turns.requests[1].chat == 'shared-thread'
    prompt = turns.requests[0].prompts[0]
    assert (prompt.actor, prompt.handle, prompt.memory_scope) == (
        _PRINCIPAL.stable_key,
        'fictional',
        'prior-memory',
    )
    assert turns.requests[2].channel == 'web'
    assert turns.requests[2].prompts[0].actor == 'client:web'


async def test_an_unlinked_asserted_user_gets_no_account_bound_memory() -> None:
    turns = StreamingTurns()
    resolver = MagicMock(spec=IdentityResolutionServiceContract)
    resolver.resolve = AsyncMock(
        return_value=AnonymousPrincipal(
            provider='http',
            realm='web',
            external_user_id='app-user-1',
        ),
    )
    async with client_for(turns, resolver) as client:
        response = await client.post(
            '/v1/chat',
            headers=headers(),
            json={'message': 'hello'},
        )
    assert response.status_code == 200
    prompt = turns.requests[0].prompts[0]
    assert prompt.account is None
    assert prompt.handle is None
    assert prompt.memory_scope is None


@pytest.mark.parametrize('asserted', ['', ' ', 'x' * 257])
async def test_invalid_asserted_identity_is_rejected(asserted: str) -> None:
    resolver = MagicMock(spec=IdentityResolutionServiceContract)
    async with client_for(StreamingTurns(), resolver) as client:
        response = await client.post(
            '/v1/chat',
            headers=headers(asserted),
            json={'message': 'hello'},
        )
    assert response.status_code == 400
    resolver.resolve.assert_not_called()


async def test_claim_locks_memory_and_target_to_the_invite() -> None:
    management = MagicMock(spec=IdentityManagementServiceContract)
    management.claim = AsyncMock(return_value=Ok(_PRINCIPAL))
    resolver = MagicMock(spec=IdentityResolutionServiceContract)
    async with client_for(StreamingTurns(), resolver, management) as client:
        response = await client.post(
            '/v1/identity/claim',
            headers=headers(),
            json={'code': 'fictional-code', 'handle': 'fictional'},
        )
        tampered = await client.post(
            '/v1/identity/claim',
            headers=headers(),
            json={
                'code': 'fictional-code',
                'handle': 'fictional',
                'memory_scope': 'another-person',
            },
        )
        missing = await client.post(
            '/v1/identity/claim',
            headers=headers(None),
            json={'code': 'fictional-code', 'handle': 'fictional'},
        )
    assert response.status_code == 200
    assert response.json()['handle'] == 'fictional'
    assert tampered.status_code == 422
    assert missing.status_code == 400
    management.claim.assert_awaited_once_with(
        'http',
        'web',
        'app-user-1',
        'fictional',
        credential='fictional-code',
        display_name=None,
    )


async def test_invalid_link_is_a_client_error_and_requires_asserted_identity() -> None:
    management = MagicMock(spec=IdentityLinkingServiceContract)
    management.link = AsyncMock(
        return_value=Err(InvalidIdentityLinkError('realm_mismatch')),
    )
    async with client_for(
        StreamingTurns(),
        MagicMock(spec=IdentityResolutionServiceContract),
        management,
    ) as client:
        response = await client.post(
            '/v1/identity/link',
            headers=headers(),
            json={'code': 'fictional-code'},
        )
        missing = await client.post(
            '/v1/identity/link',
            headers=headers(None),
            json={'code': 'fictional-code'},
        )
    assert response.status_code == 400
    assert missing.status_code == 400
    management.link.assert_awaited_once_with(
        'fictional-code',
        'http',
        'web',
        'app-user-1',
    )
