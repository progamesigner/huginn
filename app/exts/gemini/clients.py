from base64 import b64decode
from binascii import Error as Base64Error
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, ValidationError

from lib.http import HTTPClientContract

from .exceptions import GeminiGenerationError

_OVERHEAD = 64 * 1024


class GeminiValue(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)


class InlineData(GeminiValue):
    mime_type: Annotated[str, Field(alias='mimeType')]
    data: Annotated[str, Field()]


class Part(GeminiValue):
    text: Annotated[str | None, Field()] = None
    inline_data: Annotated[InlineData | None, Field(alias='inlineData')] = None
    thought: Annotated[bool, Field()] = False


class Content(GeminiValue):
    parts: Annotated[tuple[Part, ...], Field()] = ()


class Candidate(GeminiValue):
    content: Annotated[Content | None, Field()] = None
    finish_reason: Annotated[str | None, Field(alias='finishReason')] = None


class PromptFeedback(GeminiValue):
    block_reason: Annotated[str | None, Field(alias='blockReason')] = None


class GenerationResponse(GeminiValue):
    candidates: Annotated[tuple[Candidate, ...], Field()] = ()
    prompt_feedback: Annotated[PromptFeedback | None, Field(alias='promptFeedback')] = (
        None
    )


@dataclass(frozen=True)
class Generation:
    data: bytes
    mime: str
    text: str | None


class GeminiClient:
    def __init__(
        self,
        http: HTTPClientContract,
        api_key: SecretStr,
        base_url: str,
        model: str,
        max_bytes: int,
    ) -> None:
        self.http = http
        self.headers = {'x-goog-api-key': api_key.get_secret_value()}
        self.base_url = base_url
        self.model = model.removeprefix('models/')
        self.max_bytes = max_bytes

    async def generate(
        self,
        text: str,
        config: dict[str, JsonValue],
        mime_prefix: str,
    ) -> Generation:
        limit = 4 * ((self.max_bytes + 2) // 3) + _OVERHEAD
        async with (
            self.http.connect(self.base_url, self.headers) as client,
            client.stream(
                'POST',
                f'/v1beta/models/{quote(self.model, safe="")}:generateContent',
                json={
                    'contents': [{'role': 'user', 'parts': [{'text': text}]}],
                    'generationConfig': config,
                },
                timeout=120.0,
            ) as response,
        ):
            declared = response.headers.get('content-length', '')
            if declared.isdigit() and int(declared) > limit:
                raise GeminiGenerationError('the response exceeds the size limit')
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > limit:
                    raise GeminiGenerationError('the response exceeds the size limit')
                body.extend(chunk)
            if response.is_client_error:
                raise GeminiGenerationError(
                    f'Gemini refused the request ({response.status_code})',
                )
            response.raise_for_status()
        try:
            result = GenerationResponse.model_validate_json(body)
        except ValidationError as error:
            raise GeminiGenerationError(
                'Gemini returned an invalid response',
            ) from error
        if result.prompt_feedback and result.prompt_feedback.block_reason:
            raise GeminiGenerationError(
                f'Gemini blocked the prompt ({result.prompt_feedback.block_reason})',
            )
        candidate = result.candidates[0] if result.candidates else None
        if candidate and candidate.finish_reason not in (None, 'STOP'):
            raise GeminiGenerationError(
                f'Gemini did not finish generation ({candidate.finish_reason})',
            )
        parts = candidate.content.parts if candidate and candidate.content else ()
        description = '\n'.join(
            part.text for part in parts if part.text and not part.thought
        )
        for part in parts:
            inline = part.inline_data
            if (
                part.thought
                or inline is None
                or not inline.mime_type.startswith(mime_prefix)
            ):
                continue
            if len(inline.data) > 4 * ((self.max_bytes + 2) // 3):
                raise GeminiGenerationError(
                    'the generated media exceeds the size limit',
                )
            try:
                data = b64decode(inline.data, validate=True)
            except (Base64Error, ValueError) as error:
                raise GeminiGenerationError(
                    'Gemini returned invalid base64 media',
                ) from error
            if not data or len(data) > self.max_bytes:
                raise GeminiGenerationError(
                    'the generated media is empty or exceeds the size limit',
                )
            return Generation(
                data=data,
                mime=inline.mime_type,
                text=description or None,
            )
        reason = candidate.finish_reason if candidate else 'no candidates'
        raise GeminiGenerationError(
            f'Gemini returned no {mime_prefix.rstrip("/")} ({reason})',
        )
