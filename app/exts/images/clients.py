from base64 import b64decode
from dataclasses import dataclass
from typing import Annotated

from httpx2 import Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from lib.http import HTTPClientContract

from .exceptions import ImageGenerationError

_GENERATION_TIMEOUT = 120.0
_DEFAULT_MAX_BYTES = 20 * 1024 * 1024
_RESPONSE_OVERHEAD = 64 * 1024


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
    max_bytes: int

    def __init__(
        self,
        http: HTTPClientContract,
        api_key: SecretStr,
        base_url: str,
        model: str,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self.http = http
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key.get_secret_value()}'}
        self.model = model
        self.max_bytes = max_bytes

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

        async with (
            self.http.connect(self.base_url, self.headers) as client,
            client.stream(
                'POST',
                '/v1/images/generations',
                json=payload,
                timeout=_GENERATION_TIMEOUT,
            ) as response,
        ):
            if response.is_client_error:
                raise ImageGenerationError(
                    _refusal(
                        await _read_capped(response, _RESPONSE_OVERHEAD),
                        response.status_code,
                    ),
                )

            response.raise_for_status()
            body = await _read_capped(response, self._response_limit())
            images = GenerationResponse.model_validate_json(body).data
            if not images:
                raise ImageGenerationError('the vendor returned no image')

        image = images[0]
        if image.b64_json is not None:
            if len(image.b64_json) > 4 * ((self.max_bytes + 2) // 3):
                raise ImageGenerationError('the image exceeds the size limit')
            data = b64decode(image.b64_json)
            if len(data) > self.max_bytes:
                raise ImageGenerationError('the image exceeds the size limit')
            return Generation(
                data=data,
                revised_prompt=image.revised_prompt,
            )
        if image.url is not None:
            return Generation(
                data=await self._download(image.url),
                revised_prompt=image.revised_prompt,
            )

        raise ImageGenerationError('the vendor returned neither bytes nor a link')

    def _response_limit(self) -> int:
        encoded = 4 * ((self.max_bytes + 2) // 3)
        return encoded + _RESPONSE_OVERHEAD

    # NOTE: the link points at the vendor's storage, not its api, so the key stays home
    async def _download(self, url: str) -> bytes:
        if not url.startswith('https://'):
            raise ImageGenerationError('the vendor linked the image over plain http')

        async with (
            self.http.connect() as client,
            client.stream(
                'GET',
                url,
                timeout=_GENERATION_TIMEOUT,
            ) as response,
        ):
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > self.max_bytes:
                    raise ImageGenerationError('the image exceeds the size limit')
                body.extend(chunk)
            return bytes(body)


def _refusal(content: bytes, status: int) -> str:
    try:
        message = ErrorResponse.model_validate_json(content).error.message
    except ValueError:
        message = ''

    detail = f': {message}' if message else ''
    return f'the vendor refused the request ({status}){detail}'


async def _read_capped(response: Response, limit: int) -> bytes:
    declared = response.headers.get('content-length')
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise ImageGenerationError('the response exceeds the size limit')

    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > limit:
            raise ImageGenerationError('the response exceeds the size limit')
        body.extend(chunk)
    return bytes(body)
