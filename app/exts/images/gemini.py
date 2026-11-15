from pydantic import SecretStr

from app.exts.gemini.clients import GeminiClient
from app.exts.gemini.exceptions import GeminiGenerationError
from lib.http import HTTPClientContract

from .clients import Generation
from .exceptions import ImageGenerationError

_ASPECT_RATIOS = {'1024x1024': '1:1', '1536x1024': '3:2', '1024x1536': '2:3'}


class GeminiImagesClient:
    def __init__(
        self,
        http: HTTPClientContract,
        api_key: SecretStr,
        base_url: str,
        model: str,
        max_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self.client = GeminiClient(http, api_key, base_url, model, max_bytes)

    async def generate(
        self,
        prompt: str,
        size: str,
        quality: str | None = None,
    ) -> Generation:
        try:
            result = await self.client.generate(
                prompt,
                {
                    'responseModalities': ['TEXT', 'IMAGE'],
                    'imageConfig': {'aspectRatio': _ASPECT_RATIOS[size]},
                },
                'image/',
            )
        except GeminiGenerationError as error:
            raise ImageGenerationError(str(error)) from error
        return Generation(data=result.data, revised_prompt=result.text)
