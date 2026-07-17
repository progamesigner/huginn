from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
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
    environment: Annotated[Environment, Field()] = 'dev'
    providers: Annotated[tuple[ProviderConfig, ...], Field()] = ()

    def running_in_mode(self, *modes: Environment) -> bool:
        return self.environment in modes
