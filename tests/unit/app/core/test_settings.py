import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import SettingsConfigDict

from app.core.settings import HuginnSettings
from lib.llm import ProviderConfig


class IsolatedSettings(HuginnSettings):
    model_config = SettingsConfigDict(env_file=None, extra='ignore')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('HUGINN_ENVIRONMENT', raising=False)


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


def test_database_url_defaults_to_none() -> None:
    settings = IsolatedSettings()

    assert settings.database_url is None


def test_database_url_accepts_a_full_dsn() -> None:
    settings = IsolatedSettings(
        database_url='postgresql+asyncpg://scout:hunter2@db.internal:5432/huginn'
    )

    assert settings.database_url is not None
    dsn = settings.database_url.get_secret_value()
    assert str(dsn) == 'postgresql+asyncpg://scout:hunter2@db.internal:5432/huginn'


def test_database_url_masks_the_password_in_str_and_repr() -> None:
    settings = IsolatedSettings(
        database_url='postgresql+asyncpg://scout:hunter2@db.internal:5432/huginn'
    )

    assert 'hunter2' not in str(settings.database_url)
    assert 'hunter2' not in repr(settings.database_url)
    assert 'hunter2' not in settings.model_dump_json()


def test_database_url_composes_from_discrete_parts() -> None:
    settings = IsolatedSettings(
        database_host='db.internal',
        database_port=6543,
        database_user='scout',
        database_password=SecretStr('hunter2'),
        database_name='huginn',
    )

    assert settings.database_url is not None
    dsn = settings.database_url.get_secret_value()
    assert str(dsn) == 'postgresql+asyncpg://scout:hunter2@db.internal:6543/huginn'


def test_database_url_wins_over_discrete_parts() -> None:
    settings = IsolatedSettings(
        database_url='postgresql+asyncpg://explicit:pw@explicit-host:5432/explicit',
        database_host='ignored-host',
    )

    assert settings.database_url is not None
    dsn = settings.database_url.get_secret_value()
    assert str(dsn) == 'postgresql+asyncpg://explicit:pw@explicit-host:5432/explicit'


def test_database_url_env_vars_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HUGINN_DATABASE_HOST', 'db.internal')
    monkeypatch.setenv('HUGINN_DATABASE_USER', 'scout')
    monkeypatch.setenv('HUGINN_DATABASE_PASSWORD', 'hunter2')
    monkeypatch.setenv('HUGINN_DATABASE_NAME', 'huginn')

    settings = IsolatedSettings()

    assert settings.database_url is not None
    dsn = settings.database_url.get_secret_value()
    assert str(dsn) == 'postgresql+asyncpg://scout:hunter2@db.internal:5432/huginn'
