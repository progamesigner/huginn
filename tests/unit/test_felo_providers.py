from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

from app.core.application import App
from app.exts.felo.client import FeloClient
from app.exts.felo.providers import FeloExtProvider
from app.exts.felo.settings import FeloExtSettings

_PAYLOAD = {
    'status': 'ok',
    'data': {'answer': 'An answer.', 'resources': []},
}


@pytest.fixture(autouse=True)
def felo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EXT_FELO_API_KEY', 'felo-key')
    monkeypatch.delenv('EXT_FELO_BASE_URL', raising=False)


async def test_provider_binds_settings_and_client() -> None:
    app = App().build(FeloExtProvider())

    try:
        assert app.bound(FeloExtSettings)
        assert app.bound(FeloClient)
    finally:
        await app.dispose()


async def test_client_is_shared() -> None:
    app = App().build(FeloExtProvider())

    try:
        assert app.resolve(FeloClient) is app.resolve(FeloClient)
    finally:
        await app.dispose()


@respx.mock
async def test_client_searches_configured_endpoint_with_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('EXT_FELO_BASE_URL', 'https://proxy.test')
    route = respx.post('https://proxy.test/v2/chat').mock(
        return_value=httpx.Response(200, json=_PAYLOAD),
    )
    app = App().build(FeloExtProvider())

    try:
        await app.resolve(FeloClient).search('hi')
    finally:
        await app.dispose()

    assert route.called
    assert route.calls.last.request.headers['Authorization'] == 'Bearer felo-key'


async def test_start_fails_fast_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('EXT_FELO_API_KEY', raising=False)
    app = App().build(FeloExtProvider())

    with pytest.raises(ValidationError):
        await app.start()
