from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, PostgresDsn, Secret, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lib.llm import ProviderConfig

type Environment = Literal['dev', 'staging', 'prod']


class HuginnSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='HUGINN_',
        env_file='.env',
        extra='ignore',
    )

    config_file: Annotated[Path, Field()] = Path('huginn.yaml')
    database_url: Annotated[Secret[PostgresDsn] | None, Field()] = None
    database_host: Annotated[str | None, Field()] = None
    database_port: Annotated[int, Field()] = 5432
    database_user: Annotated[str | None, Field()] = None
    database_password: Annotated[SecretStr | None, Field()] = None
    database_name: Annotated[str | None, Field()] = None
    environment: Annotated[Environment, Field()] = 'dev'
    identity_link_secret: Annotated[SecretStr | None, Field()] = None
    providers: Annotated[tuple[ProviderConfig, ...], Field()] = ()

    def running_in_mode(self, *modes: Environment) -> bool:
        return self.environment in modes

    @model_validator(mode='after')
    def _compose_database_url(self) -> Self:
        if self.database_url is not None or self.database_host is None:
            return self

        dsn = PostgresDsn.build(
            scheme='postgresql+asyncpg',
            username=self.database_user,
            password=(
                self.database_password.get_secret_value()
                if self.database_password
                else None
            ),
            host=self.database_host,
            port=self.database_port,
            path=self.database_name,
        )
        self.database_url = Secret(dsn)
        return self
