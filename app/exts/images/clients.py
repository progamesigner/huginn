from base64 import b64decode
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from lib.http import HTTPClientContract

from .exceptions import ImageGenerationError

_GENERATION_TIMEOUT = 120.0


class ImagesValue(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)


class GeneratedImage(ImagesValue):
    b64_json: Annotated[str | None, Field()] = None
    url: Annotated[str | None, Field()] = None
    revised_prompt: Annotated[str | None, Field()] = None


class GenerationResponse(ImagesValue):
    data: Annotated[tuple[GeneratedImage, ...], Field()]


class ImageError(ImagesValue):
    message: Annotated[str, Field()] = ''


class ErrorResponse(ImagesValue):
    error: Annotated[ImageError, Field()]


@dataclass(frozen=True)
class Generation:
    data: bytes
    revised_prompt: str | None = None


class ImagesClient:
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

    async def generate(
        self,
        prompt: str,
        size: str,
        quality: str | None = None,
    ) -> Generation:
        payload = {
            'model': self.model,
            'prompt': prompt,
            'size': size,
            'n': 1,
            **({} if quality is None else {'quality': quality}),
        }

        async with self.http.connect(self.base_url, self.headers) as client:
            response = await client.post(
                '/v1/images/generations',
                json=payload,
                timeout=_GENERATION_TIMEOUT,
            )
            if response.is_client_error:
                raise ImageGenerationError(
                    _refusal(response.content, response.status_code),
                )

            images = GenerationResponse.model_validate_json(
                response.raise_for_status().content,
            ).data
            if not images:
                raise ImageGenerationError('the vendor returned no image')

        image = images[0]
        if image.b64_json is not None:
            return Generation(
                data=b64decode(image.b64_json),
                revised_prompt=image.revised_prompt,
            )
        if image.url is not None:
            return Generation(
                data=await self._download(image.url),
                revised_prompt=image.revised_prompt,
            )

        raise ImageGenerationError('the vendor returned neither bytes nor a link')

    # NOTE: the link points at the vendor's storage, not its api, so the key stays home
    async def _download(self, url: str) -> bytes:
        if not url.startswith('https://'):
            raise ImageGenerationError('the vendor linked the image over plain http')

        async with self.http.connect() as client:
            response = await client.get(url, timeout=_GENERATION_TIMEOUT)
            return response.raise_for_status().content


def _refusal(content: bytes, status: int) -> str:
    try:
        message = ErrorResponse.model_validate_json(content).error.message
    except ValueError:
        message = ''

    detail = f': {message}' if message else ''
    return f'the vendor refused the request ({status}){detail}'
