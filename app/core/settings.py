from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type Environment = Literal['dev', 'staging', 'prod']


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='HUGINN_',
        env_file='.env',
        extra='ignore',
    )

    environment: Annotated[Environment, Field()] = 'dev'

    def running_in_mode(self, *modes: Environment) -> bool:
        return self.environment in modes
