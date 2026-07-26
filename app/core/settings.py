from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, PostgresDsn, RedisDsn, Secret, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from lib.llm import ProviderConfig

type Environment = Literal['dev', 'staging', 'prod']
type ConnectionName = str | None


class HuginnSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='HUGINN_',
        env_file='.env',
        extra='ignore',
    )

    environment: Annotated[Environment, Field()] = 'dev'
    config_file: Annotated[Path, Field()] = Path('huginn.yaml')
    database_url: Annotated[Secret[PostgresDsn] | None, Field()] = None
    database_host: Annotated[str | None, Field()] = None
    database_port: Annotated[int, Field()] = 5432
    database_user: Annotated[str | None, Field()] = None
    database_password: Annotated[SecretStr | None, Field()] = None
    database_name: Annotated[str | None, Field()] = None
    redis_url: Annotated[Secret[RedisDsn] | None, Field()] = None
    redis_host: Annotated[str | None, Field()] = None
    redis_port: Annotated[int, Field()] = 6379
    redis_user: Annotated[str | None, Field()] = None
    redis_password: Annotated[SecretStr | None, Field()] = None
    redis_database: Annotated[int, Field()] = 0
    identity_link_secret: Annotated[SecretStr | None, Field()] = None
    providers: Annotated[tuple[ProviderConfig, ...], Field()] = ()

    @computed_field
    @property
    def database_urls(self) -> dict[ConnectionName, Secret[PostgresDsn]]:
        if self.database_url is not None:
            return {None: self.database_url}

        if self.database_host is None:
            return {}

        return {
            None: Secret(
                PostgresDsn.build(
                    scheme='postgresql+asyncpg',
                    username=self.database_user,
                    password=(
                        self.database_password.get_secret_value()
                        if self.database_password is not None
                        else None
                    ),
                    host=self.database_host,
                    port=self.database_port,
                    path=self.database_name,
                ),
            ),
        }

    @computed_field
    @property
    def redis_urls(self) -> dict[ConnectionName, Secret[RedisDsn]]:
        if self.redis_url is not None:
            return {None: self.redis_url}

        if self.redis_host is None:
            return {}

        return {
            None: Secret(
                RedisDsn.build(
                    scheme='redis',
                    username=self.redis_user,
                    password=(
                        self.redis_password.get_secret_value()
                        if self.redis_password is not None
                        else None
                    ),
                    host=self.redis_host,
                    port=self.redis_port,
                    path=str(self.redis_database),
                ),
            ),
        }

    def running_in_mode(self, *modes: Environment) -> bool:
        return self.environment in modes
