import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import SettingsConfigDict

from app.core.settings import HuginnSettings
from lib.llm import ProviderConfig


class IsolatedSettings(HuginnSettings):
    model_config = SettingsConfigDict(env_file=None, extra='ignore')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        'HUGINN_ENVIRONMENT',
        'HUGINN_DATABASE_URL',
        'HUGINN_DATABASE_HOST',
        'HUGINN_DATABASE_PORT',
        'HUGINN_DATABASE_USER',
        'HUGINN_DATABASE_PASSWORD',
        'HUGINN_DATABASE_NAME',
        'HUGINN_REDIS_URL',
        'HUGINN_REDIS_HOST',
        'HUGINN_REDIS_PORT',
        'HUGINN_REDIS_USER',
        'HUGINN_REDIS_PASSWORD',
        'HUGINN_REDIS_DATABASE',
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults() -> None:
    settings = IsolatedSettings()

    assert settings.environment == 'dev'
    assert settings.providers == ()


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


def test_running_in_mode_matches_any_given_mode() -> None:
    settings = IsolatedSettings(environment='staging')

    assert settings.running_in_mode('staging')
    assert settings.running_in_mode('staging', 'prod')
    assert not settings.running_in_mode('dev')


def test_providers_accepts_constructor_override() -> None:
    provider = ProviderConfig(
        name='anthropic',
        adapter='anthropic',
        api_key=SecretStr('secret'),
        models=('claude-opus-4-8',),
    )

    settings = IsolatedSettings(providers=(provider,))

    assert settings.providers == (provider,)


def test_connection_urls_default_to_empty_mappings() -> None:
    settings = IsolatedSettings()

    assert settings.database_urls == {}
    assert settings.redis_urls == {}
    assert settings.model_dump()['database_urls'] == {}
    assert settings.model_dump()['redis_urls'] == {}


def test_connection_urls_accept_direct_dsns() -> None:
    settings = IsolatedSettings(
        database_url='postgresql+asyncpg://scout:hunter2@db.internal:5432/huginn',
        redis_url='redis://cache.internal:6379/1',
    )

    assert str(settings.database_urls[None].get_secret_value()) == (
        'postgresql+asyncpg://scout:hunter2@db.internal:5432/huginn'
    )
    assert str(settings.redis_urls[None].get_secret_value()) == (
        'redis://cache.internal:6379/1'
    )


def test_connection_urls_build_from_parts() -> None:
    settings = IsolatedSettings(
        database_host='db.internal',
        database_port=6543,
        database_user='scout',
        database_password=SecretStr('hunter2'),
        database_name='huginn',
        redis_host='cache.internal',
        redis_port=6380,
        redis_user='scout',
        redis_password=SecretStr('hunter2'),
        redis_database=2,
    )

    assert str(settings.database_urls[None].get_secret_value()) == (
        'postgresql+asyncpg://scout:hunter2@db.internal:6543/huginn'
    )
    assert str(settings.redis_urls[None].get_secret_value()) == (
        'redis://scout:hunter2@cache.internal:6380/2'
    )


def test_direct_connection_urls_take_precedence_over_parts() -> None:
    settings = IsolatedSettings(
        database_url='postgresql+asyncpg://db.internal/huginn',
        database_host='ignored.internal',
        redis_url='redis://cache.internal/0',
        redis_host='ignored.internal',
    )

    assert str(settings.database_urls[None].get_secret_value()) == (
        'postgresql+asyncpg://db.internal/huginn'
    )
    assert str(settings.redis_urls[None].get_secret_value()) == (
        'redis://cache.internal:6379/0'
    )


def test_connection_settings_mask_passwords() -> None:
    settings = IsolatedSettings(
        database_password=SecretStr('hunter2'),
        redis_password=SecretStr('hunter2'),
    )

    assert 'hunter2' not in settings.model_dump_json()


def test_connection_urls_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HUGINN_DATABASE_HOST', 'db.internal')
    monkeypatch.setenv('HUGINN_DATABASE_USER', 'analyst')
    monkeypatch.setenv('HUGINN_DATABASE_PASSWORD', 'secret')
    monkeypatch.setenv('HUGINN_DATABASE_NAME', 'events')
    monkeypatch.setenv('HUGINN_REDIS_URL', 'redis://cache.internal/0')

    settings = IsolatedSettings()

    assert str(settings.database_urls[None].get_secret_value()) == (
        'postgresql+asyncpg://analyst:secret@db.internal:5432/events'
    )
    assert str(settings.redis_urls[None].get_secret_value()) == (
        'redis://cache.internal:6379/0'
    )
