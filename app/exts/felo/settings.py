from typing import Annotated

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeloExtSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='EXT_FELO_',
        env_file='.env',
        extra='ignore',
    )

    api_key: Annotated[SecretStr, Field()]
    base_url: Annotated[HttpUrl, Field()] = HttpUrl('https://openapi.felo.ai')
