from typing import Annotated, override

import httpx
from fastapi import FastAPI
from starlette.types import ASGIApp

from app.contracts.auth import ApiClient, AuthContract
from app.core.dependencies import Authenticated
from app.core.exceptions import AppError
from app.core.renderers import render_app_error, render_unexpected_error
from lib.container import Container
from lib.router import Depends, Router, RouterManager


class FakeAuth(AuthContract):
    tokens: dict[str, ApiClient]
    offered: list[str]

    def __init__(self, tokens: dict[str, ApiClient]) -> None:
        self.tokens = tokens
        self.offered = []

    @override
    def authenticate(self, credential: str) -> ApiClient | None:
        self.offered.append(credential)
        return self.tokens.get(credential)


def build_app(auth: AuthContract) -> ASGIApp:
    container = Container()
    container.bind(AuthContract, instance=auth)

    router = Router()

    @router.get('/whoami')
    async def whoami(
        client: Annotated[ApiClient, Depends(Authenticated())],
    ) -> dict[str, str]:
        return {'key': client.key}

    manager = RouterManager(
        exception_handlers={
            AppError: render_app_error,
            Exception: render_unexpected_error,
        },
    )
    manager.include(router)
    app = manager.app
    assert isinstance(app, FastAPI)
    app.state.container = container
    return app


async def fetch(app: ASGIApp, headers: dict[str, str]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        return await client.get('/whoami', headers=headers)


def client(key: str = 'web') -> ApiClient:
    return ApiClient(key=key, agents=frozenset({'scout'}))


async def test_a_bearer_token_reaches_the_endpoint_as_its_client() -> None:
    auth = FakeAuth({'web-token': client()})

    response = await fetch(build_app(auth), {'Authorization': 'Bearer web-token'})

    assert response.status_code == 200
    assert response.json() == {'key': 'web'}
    assert auth.offered == ['web-token']


async def test_the_scheme_is_read_case_insensitively() -> None:
    auth = FakeAuth({'web-token': client()})

    response = await fetch(build_app(auth), {'Authorization': 'bearer web-token'})

    assert response.status_code == 200


async def test_an_unknown_token_is_unauthorized() -> None:
    auth = FakeAuth({'web-token': client()})

    response = await fetch(build_app(auth), {'Authorization': 'Bearer wrong'})

    assert response.status_code == 401
    assert response.json()['error'] == 'UNAUTHORIZED'
    assert response.headers['WWW-Authenticate'] == 'Bearer'


async def test_a_missing_header_is_unauthorized_without_asking_the_directory() -> None:
    auth = FakeAuth({'web-token': client()})

    response = await fetch(build_app(auth), {})

    assert response.status_code == 401
    assert auth.offered == []


async def test_another_scheme_never_reaches_the_directory() -> None:
    auth = FakeAuth({'web-token': client()})

    response = await fetch(build_app(auth), {'Authorization': 'Basic d2ViOnRva2Vu'})

    assert response.status_code == 401
    assert auth.offered == []


async def test_an_empty_bearer_credential_never_reaches_the_directory() -> None:
    # NOTE: 'Bearer ' with nothing after it is what an unset frontend env var sends
    auth = FakeAuth({'': client()})

    response = await fetch(build_app(auth), {'Authorization': 'Bearer  '})

    assert response.status_code == 401
    assert auth.offered == []


async def test_the_unauthorized_message_says_nothing_about_the_token() -> None:
    auth = FakeAuth({'web-token': client()})

    missing = await fetch(build_app(auth), {})
    unknown = await fetch(build_app(auth), {'Authorization': 'Bearer wrong'})

    assert missing.json()['message'] == unknown.json()['message']
    assert 'wrong' not in unknown.text
