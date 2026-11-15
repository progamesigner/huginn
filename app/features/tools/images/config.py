import warnings
from typing import Annotated, Literal, Self

from pydantic import Field, HttpUrl, SecretStr, model_validator

from app.features.tools.config import ToolConfig

type ImageSize = Literal['1024x1024', '1536x1024', '1024x1536']


class ImageToolConfig(ToolConfig):
    kind: Annotated[Literal['images'], Field()]
    vendor: Annotated[Literal['openai', 'gemini'], Field()] = 'openai'
    model: Annotated[str, Field()]
    api_key: Annotated[SecretStr, Field()]
    base_url: Annotated[HttpUrl, Field()] = HttpUrl('https://api.openai.com')
    size: Annotated[ImageSize, Field()] = '1024x1024'
    quality: Annotated[str | None, Field()] = None
    max_bytes: Annotated[int, Field(gt=0)] = 20 * 1024 * 1024

    @model_validator(mode='after')
    def vendor_defaults(self) -> Self:
        if self.vendor == 'gemini':
            if 'base_url' not in self.model_fields_set:
                object.__setattr__(
                    self,
                    'base_url',
                    HttpUrl('https://generativelanguage.googleapis.com'),
                )
            if self.quality is not None:
                warnings.warn(
                    'Gemini images ignore quality; use size to choose the aspect ratio',
                    UserWarning,
                    stacklevel=2,
                )
        return self
