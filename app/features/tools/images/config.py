from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr

from app.features.tools.config import ToolConfig

type ImageSize = Literal['1024x1024', '1536x1024', '1024x1536']


class ImageToolConfig(ToolConfig):
    kind: Annotated[Literal['images'], Field()]
    model: Annotated[str, Field()]
    api_key: Annotated[SecretStr, Field()]
    base_url: Annotated[HttpUrl, Field()] = HttpUrl('https://api.openai.com')
    size: Annotated[ImageSize, Field()] = '1024x1024'
    quality: Annotated[str | None, Field()] = None
    max_bytes: Annotated[int, Field(gt=0)] = 20 * 1024 * 1024
