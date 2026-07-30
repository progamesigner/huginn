from typing import Annotated

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
    model_config = ConfigDict(extra='ignore', frozen=True)


class Transcription(TranscriptionValue):
    text: Annotated[str, Field()]


def filename(mime: str) -> str:
    # NOTE: openai reads the format off the upload's extension, not its content type
    return f'audio.{_EXTENSIONS.get(mime, mime.rpartition("/")[2])}'


class TranscriptionClient:
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
