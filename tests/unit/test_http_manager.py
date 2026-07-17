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


async def test_connect_closes_client_on_exit() -> None:
    manager = HTTPClientManager()

    async with manager.connect('https://api.test') as client:
        assert not client.is_closed

    assert client.is_closed
