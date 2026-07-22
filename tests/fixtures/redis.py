from collections.abc import Iterator

import pytest

from app.core.settings import HuginnSettings

_IMAGE = 'redis:8'


@pytest.fixture(scope='session')
def redis_url() -> Iterator[str]:
    configured = HuginnSettings().redis_urls.get(None)
    if configured is not None:
        yield str(configured.get_secret_value())
        return

    try:
        from testcontainers.redis import RedisContainer
    except ImportError:
        pytest.skip('set HUGINN_REDIS_URL or add the testcontainers dev dependency')

    try:
        container = RedisContainer(_IMAGE)
        with container:
            host = container.get_container_host_ip()
            port = container.get_exposed_port(6379)
            yield f'redis://{host}:{port}/0'
    except Exception as error:
        pytest.skip(
            f'Docker unavailable; set HUGINN_REDIS_URL to run redis tests ({error})'
        )
