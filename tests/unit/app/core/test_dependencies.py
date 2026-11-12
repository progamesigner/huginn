from typing import Annotated

import httpx2
from fastapi import Depends, FastAPI

from app.core.dependencies import Provide
from lib.container import Container


class Greeter:
    def __init__(self, name: str) -> None:
        self.name = name


class Other:
    pass


def build_greeting_app(container: Container) -> FastAPI:
    app = FastAPI()
    app.state.container = container

    @app.get('/greet')
    async def greet(
        greeter: Annotated[Greeter, Depends(Provide(Greeter))],
    ) -> dict[str, str]:
        return {'name': greeter.name}

    return app


def test_provide_identity_follows_binding() -> None:
    assert Provide(Greeter) == Provide(Greeter)
    assert hash(Provide(Greeter)) == hash(Provide(Greeter))
    assert Provide(Greeter) != Provide(Other)


async def test_provide_resolves_from_app_state_container() -> None:
    container = Container()
    container.bind(Greeter, instance=Greeter('from-container'))
    app = build_greeting_app(container)

    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        response = await client.get('/greet')

    assert response.status_code == 200
    assert response.json() == {'name': 'from-container'}


async def test_container_use_swaps_endpoint_dependency() -> None:
    container = Container()
    container.bind(Greeter, instance=Greeter('real'))
    app = build_greeting_app(container)

    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url='http://huginn',
    ) as client:
        with container.use(Greeter, Greeter('fake')):
            overridden = await client.get('/greet')
        restored = await client.get('/greet')

    assert overridden.json() == {'name': 'fake'}
    assert restored.json() == {'name': 'real'}
