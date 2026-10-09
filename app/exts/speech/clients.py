from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from lib.http import HTTPClientContract

from .exceptions import SpeechSynthesisError

_SYNTHESIS_TIMEOUT = 120.0


class SpeechValue(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)


class SpeechError(SpeechValue):
    message: Annotated[str, Field()] = ''


class ErrorResponse(SpeechValue):
    error: Annotated[SpeechError, Field()]


class SpeechClient:
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
        self.http = http
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key.get_secret_value()}'}
        self.model = model

    # NOTE: asking for opus gets ogg-wrapped opus back, which is what a voice note is,
    # so nothing downstream has to transcode
    async def synthesize(self, text: str, voice: str, response_format: str) -> bytes:
        async with self.http.connect(self.base_url, self.headers) as client:
            response = await client.post(
                '/v1/audio/speech',
                json={
                    'model': self.model,
                    'input': text,
                    'voice': voice,
                    'response_format': response_format,
                },
                timeout=_SYNTHESIS_TIMEOUT,
            )
            if response.is_client_error:
                raise SpeechSynthesisError(
                    _refusal(response.content, response.status_code),
                )

            return response.raise_for_status().content


def _refusal(content: bytes, status: int) -> str:
    try:
        message = ErrorResponse.model_validate_json(content).error.message
    except ValueError:
        message = ''

    detail = f': {message}' if message else ''
    return f'the vendor refused the request ({status}){detail}'
