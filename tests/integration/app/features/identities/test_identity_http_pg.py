from dataclasses import replace
from datetime import timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from app.core.config.registry import HuginnConfig
from app.features.agents.contracts import TurnServiceContract
from app.features.blobs.contracts import BlobServiceContract
from app.features.channels.config import CHANNELS, ChannelConfig, ChannelConfigList
from app.features.identities.registration import RegistrationService
from app.features.identities.repositories import IdentityRepository
from app.features.identities.tokens import IdentityTokenCodec
from bin.server import build
from lib.database import DatabaseContract, DatabaseManager
from lib.guid import GUIDManager
from lib.mcp import MCPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract
from lib.temporal import ClockManager
from tests.helpers.blobs import MemoryBlobs
from tests.helpers.redis import FakeClient, make_redis
from tests.unit.app.features.messages.test_native_chat import (
    StreamingTurns,
    api_key,
    make_config,
)

pytestmark = pytest.mark.integration


@pytest.mark.usefixtures('_migrated')
async def test_http_registration_display_name_link_and_chat_share_real_identity(
    database_url: str,
) -> None:
    database = DatabaseManager({None: database_url})
    tokens = IdentityTokenCodec(b'i' * 32)
    suffix = uuid4().hex
    handle, memory = 'person-' + suffix, 'prior-' + suffix
    external = 'app-person-' + suffix
    turns = StreamingTurns()
    config = make_config(api_key(agent='scout'))
    config = replace(
        config,
        sections={
            **config.sections,
            CHANNELS.key: ChannelConfigList(
                {
                    'other-bot': ChannelConfig(kind='telegram'),
                },
            ),
        },
    )
    async with database.transaction() as session:
        invitation = await RegistrationService(
            IdentityRepository(session, GUIDManager(), ClockManager()),
            tokens,
        ).issue(
            'http',
            'web',
            timedelta(minutes=10),
            'test-operator',
            existing_scope=memory,
        )
    application = build()
    try:
        with (
            application.use(DatabaseContract, database),
            application.use(IdentityTokenCodec, tokens),
            application.use(BlobServiceContract, MemoryBlobs()),
            application.use(HuginnConfig, config),
            application.use(MCPClientContract, MagicMock(spec=MCPClientContract)),
            application.use(RedisContract, make_redis(FakeClient())),
            application.use(TurnServiceContract, turns),
        ):
            app = application.resolve(RouterContract).app
            assert isinstance(app, FastAPI)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app),
                    base_url='http://huginn',
                ) as client:
                    headers = {
                        'Authorization': 'Bearer web-key',
                        'X-Huginn-User': external,
                    }
                    claimed = await client.post(
                        '/v1/identity/claim',
                        headers=headers,
                        json={'code': invitation.token, 'handle': handle},
                    )
                    assert claimed.status_code == 200
                    account = claimed.json()['account']
                    renamed = await client.patch(
                        '/v1/identity',
                        headers=headers,
                        json={'display_name': 'Fictional Display Name'},
                    )
                    assert renamed.status_code == 200
                    assert renamed.json()['display_name'] == 'Fictional Display Name'
                    assert renamed.json()['account'] == account
                    link = await client.post(
                        '/v1/identity/invite',
                        headers=headers,
                        json={'target': 'http:web'},
                    )
                    assert link.status_code == 200
                    other = {**headers, 'X-Huginn-User': 'second-' + external}
                    linked = await client.post(
                        '/v1/identity/link',
                        headers=other,
                        json={'code': link.json()['code']},
                    )
                    assert linked.status_code == 200
                    assert linked.json()['account'] == account
                    response = await client.post(
                        '/v1/chat',
                        headers=other,
                        json={'message': 'hello'},
                    )
                    assert response.status_code == 200
                    principal = turns.requests[0].prompts[0]
                    assert (
                        principal.account,
                        principal.handle,
                        principal.memory_scope,
                    ) == (account, handle, memory)
                    wrong_target = await client.post(
                        '/v1/identity/invite',
                        headers=headers,
                        json={'target': 'absent-bot'},
                    )
                    assert wrong_target.status_code == 400
                    unlinked = await client.post(
                        '/v1/identity/invite',
                        headers={**headers, 'X-Huginn-User': 'anonymous-' + external},
                        json={'target': 'other-bot'},
                    )
                    assert unlinked.status_code == 400
    finally:
        await database.dispose()
