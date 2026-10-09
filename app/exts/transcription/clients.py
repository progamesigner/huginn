from abc import ABC, abstractmethod
from base64 import b64encode
from typing import Annotated, override

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from lib.http import HTTPClientContract

_EXTENSIONS = {
    'audio/mp4': 'm4a',
    'audio/mpeg': 'mp3',
    'audio/ogg': 'ogg',
    'audio/x-m4a': 'm4a',
    'audio/wav': 'wav',
    'audio/webm': 'webm',
}


class TranscriptionValue(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)


class Transcription(TranscriptionValue):
    text: Annotated[str, Field()]


def filename(mime: str) -> str:
    # NOTE: openai reads the format off the upload's extension, not its content type
    return f'audio.{_EXTENSIONS.get(mime, mime.rpartition("/")[2])}'


class TranscriptionClientContract(ABC):
    @abstractmethod
    async def transcribe(self, data: bytes, mime: str) -> str: ...


class TranscriptionClient(TranscriptionClientContract):
    default_base_url = 'https://api.openai.com'

    http: HTTPClientContract
    base_url: str
    headers: dict[str, str]
    model: str

    def __init__(
        self,
        http: HTTPClientContract,
        api_key: SecretStr,
        base_url: str,
        model: str,
    ) -> None:
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key.get_secret_value()}'}
        self.http = http
        self.model = model

    @override
    async def transcribe(self, data: bytes, mime: str) -> str:
        async with self.http.connect(self.base_url, self.headers) as client:
            response = await client.post(
                '/v1/audio/transcriptions',
                data={'model': self.model},
                files={'file': (filename(mime), data, mime)},
            )
            return Transcription.model_validate_json(
                response.raise_for_status().content,
            ).text


class GeminiContent(TranscriptionValue):
    type: Annotated[str, Field()]
    text: Annotated[str | None, Field()] = None


class GeminiStep(TranscriptionValue):
    type: Annotated[str, Field()]
    content: Annotated[tuple[GeminiContent, ...], Field()] = ()


class GeminiInteraction(TranscriptionValue):
    steps: Annotated[tuple[GeminiStep, ...], Field()] = ()

    def text(self) -> str:
        return ''.join(
            part.text or ''
            for step in self.steps
            if step.type == 'model_output'
            for part in step.content
            if part.type == 'text'
        )


# NOTE: the interactions api takes the audio inline, so a voice note needs no
# separate upload to the files api first
class GeminiTranscriptionClient(TranscriptionClientContract):
    default_base_url = 'https://generativelanguage.googleapis.com'

    http: HTTPClientContract
    base_url: str
    headers: dict[str, str]
    model: str
    mode: str
    languages: tuple[str, ...]

    def __init__(
        self,
        http: HTTPClientContract,
        api_key: SecretStr,
        base_url: str,
        model: str,
        mode: str = 'smart',
        languages: tuple[str, ...] = (),
    ) -> None:
        self.base_url = base_url
        self.headers = {'x-goog-api-key': api_key.get_secret_value()}
        self.http = http
        self.languages = languages
        self.mode = mode
        self.model = model

    @override
    async def transcribe(self, data: bytes, mime: str) -> str:
        async with self.http.connect(self.base_url, self.headers) as client:
            response = await client.post(
                '/v1beta/interactions',
                json={
                    'model': self.model,
                    'input': [
                        {
                            'type': 'audio',
                            'data': b64encode(data).decode('ascii'),
                            'mime_type': mime,
                        },
                    ],
                    'generation_config': {
                        'transcription_config': {
                            'mode': self.mode,
                            'language_codes': list(self.languages),
                        },
                    },
                },
            )
            return GeminiInteraction.model_validate_json(
                response.raise_for_status().content,
            ).text()


class ElevenLabsTranscriptionClient(TranscriptionClientContract):
    default_base_url = 'https://api.elevenlabs.io'

    http: HTTPClientContract
    base_url: str
    headers: dict[str, str]
    model: str
    language: str | None

    def __init__(
        self,
        http: HTTPClientContract,
        api_key: SecretStr,
        base_url: str,
        model: str,
        language: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.headers = {'xi-api-key': api_key.get_secret_value()}
        self.http = http
        self.language = language
        self.model = model

    @override
    async def transcribe(self, data: bytes, mime: str) -> str:
        fields = {'model_id': self.model}
        if self.language is not None:
            fields['language_code'] = self.language

        async with self.http.connect(self.base_url, self.headers) as client:
            response = await client.post(
                '/v1/speech-to-text',
                data=fields,
                files={'file': (filename(mime), data, mime)},
            )
            return Transcription.model_validate_json(
                response.raise_for_status().content,
            ).text
