from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr

from app.features.tools.config import ToolConfig

type SpeechFormat = Literal['aac', 'flac', 'mp3', 'opus', 'wav']


class SpeechToolConfig(ToolConfig):
    kind: Annotated[Literal['speech'], Field()]
    model: Annotated[str, Field()]
    voice: Annotated[str, Field()]
    api_key: Annotated[SecretStr, Field()]
    base_url: Annotated[HttpUrl, Field()] = HttpUrl('https://api.openai.com')
    format: Annotated[SpeechFormat, Field()] = 'opus'
    max_chars: Annotated[int, Field(gt=0, le=4096)] = 4096
