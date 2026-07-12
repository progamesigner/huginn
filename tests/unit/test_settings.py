import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from app.core.settings import AppSettings


class IsolatedSettings(AppSettings):
    model_config = SettingsConfigDict(env_file=None, extra='ignore')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('HUGINN_ENVIRONMENT', raising=False)


def test_defaults() -> None:
    settings = IsolatedSettings()

    assert settings.environment == 'dev'


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HUGINN_ENVIRONMENT', 'prod')

    settings = IsolatedSettings()

    assert settings.environment == 'prod'


def test_constructor_kwargs_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HUGINN_ENVIRONMENT', 'prod')

    settings = IsolatedSettings(environment='staging')

    assert settings.environment == 'staging'


def test_invalid_environment_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('HUGINN_ENVIRONMENT', 'qa')

    with pytest.raises(ValidationError, match='environment'):
        IsolatedSettings()
