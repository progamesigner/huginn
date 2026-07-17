from lib.container import Disposable
from lib.http import HTTPClientManager


async def test_connect_configures_client_with_manager_defaults() -> None:
    manager = HTTPClientManager(timeout=12.0)

    async with manager.connect(
        'https://api.test',
        headers={'Authorization': 'Bearer token'},
    ) as client:
        assert str(client.base_url) == 'https://api.test'
        assert client.timeout.read == 12.0
        assert client.headers['Authorization'] == 'Bearer token'


async def test_connect_applies_configured_limits() -> None:
    manager = HTTPClientManager(max_connections=7)

    assert manager.limits.max_connections == 7

    async with manager.connect() as client:
        assert client.timeout.read == manager.timeout.read


async def test_connect_keeps_shared_pool_open_on_exit() -> None:
    manager = HTTPClientManager()
    transport = manager.transport

    async with manager.connect('https://api.test') as client:
        pass

    assert not client.is_closed
    assert manager.transport is transport


async def test_connect_routes_clients_through_shared_transport() -> None:
    manager = HTTPClientManager()

    async with manager.connect() as first, manager.connect() as second:
        assert first._transport is manager.transport
        assert second._transport is manager.transport


async def test_manager_disposes_for_container_teardown() -> None:
    manager = HTTPClientManager()

    assert isinstance(manager, Disposable)
    await manager.dispose()
