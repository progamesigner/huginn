import pytest
from pydantic import HttpUrl, SecretStr, ValidationError
from pydantic_settings import SettingsConfigDict

from app.exts.felo.settings import FeloExtSettings


class IsolatedSettings(FeloExtSettings):
    model_config = SettingsConfigDict(
        env_prefix='EXT_FELO_',
        env_file=None,
        extra='ignore',
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('EXT_FELO_API_KEY', raising=False)
    monkeypatch.delenv('EXT_FELO_BASE_URL', raising=False)


def test_api_key_is_required() -> None:
    with pytest.raises(ValidationError, match='api_key'):
        IsolatedSettings()


def test_base_url_defaults_to_public_endpoint() -> None:
    settings = IsolatedSettings(api_key=SecretStr('felo-key'))

    assert settings.base_url == HttpUrl('https://openapi.felo.ai')


def test_reads_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EXT_FELO_API_KEY', 'from-env')
    monkeypatch.setenv('EXT_FELO_BASE_URL', 'https://proxy.test')

    settings = IsolatedSettings()

    assert settings.api_key.get_secret_value() == 'from-env'
    assert settings.base_url == HttpUrl('https://proxy.test')
