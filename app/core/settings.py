from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='HUGINN_',
        env_file='.env',
        extra='ignore',
    )

    environment: Annotated[Literal['dev', 'staging', 'prod'], Field(default='dev')]
