from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr

from app.features.tools.config import ToolConfig


class WebSearchToolConfig(ToolConfig):
    kind: Annotated[Literal['websearch'], Field()]
    api_key: Annotated[SecretStr, Field()]
    base_url: Annotated[HttpUrl, Field()] = HttpUrl('https://openapi.felo.ai')
