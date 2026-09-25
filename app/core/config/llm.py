from typing import Annotated, Literal

from pydantic import Field, SecretStr

from app.core.config.schema import ConfigValue
from lib.llm import Adapter


class ProviderConfig(ConfigValue):
    type: Annotated[Literal['llm'], Field()] = 'llm'
    adapter: Annotated[Adapter, Field()]
    api_key: Annotated[SecretStr, Field()]
    models: Annotated[tuple[str, ...], Field()]
    base_url: Annotated[str | None, Field()] = None
    cache_control: Annotated[tuple[str, ...], Field()] = ()
    vision: Annotated[tuple[str, ...], Field()] = ()
    documents: Annotated[tuple[str, ...], Field()] = ()
