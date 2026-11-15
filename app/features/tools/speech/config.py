from typing import Annotated, Literal, Self

from pydantic import Field, HttpUrl, SecretStr, model_validator

from app.features.tools.config import ToolConfig
from lib.media.encoding import opus_encoder

type SpeechFormat = Literal['aac', 'flac', 'mp3', 'opus', 'wav']


class SpeechToolConfig(ToolConfig):
    kind: Annotated[Literal['speech'], Field()]
    vendor: Annotated[Literal['openai', 'gemini'], Field()] = 'openai'
    model: Annotated[str, Field()]
    voice: Annotated[str, Field()]
    api_key: Annotated[SecretStr, Field()]
    base_url: Annotated[HttpUrl, Field()] = HttpUrl('https://api.openai.com')
    format: Annotated[SpeechFormat, Field()] = 'opus'
    max_chars: Annotated[int, Field(gt=0, le=4096)] = 4096

    @model_validator(mode='after')
    def vendor_defaults(self) -> Self:
        if self.vendor == 'gemini':
            if 'base_url' not in self.model_fields_set:
                object.__setattr__(
                    self,
                    'base_url',
                    HttpUrl('https://generativelanguage.googleapis.com'),
                )
            if self.format != 'opus':
                raise ValueError('Gemini speech supports only opus output')
            opus_encoder()
            if not self.voice.strip():
                raise ValueError('Gemini voice must not be blank')
        return self
