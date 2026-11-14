from dataclasses import dataclass

from app.exts.images.clients import ImagesClient
from app.exts.images.exceptions import ImageGenerationError
from app.exts.images.gemini import GeminiImagesClient
from app.features.tools.exceptions import ToolError
from lib.http import HTTPClientContract
from lib.media import sniff

from .config import ImageToolConfig


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    mime: str
    revised_prompt: str | None = None


class ImageRepository:
    http: HTTPClientContract

    def __init__(self, http: HTTPClientContract) -> None:
        self.http = http

    async def generate(
        self,
        config: ImageToolConfig,
        prompt: str,
        size: str,
    ) -> GeneratedImage:
        client_type = GeminiImagesClient if config.vendor == 'gemini' else ImagesClient
        client = client_type(
            http=self.http,
            api_key=config.api_key,
            base_url=str(config.base_url),
            model=config.model,
            max_bytes=config.max_bytes,
        )
        try:
            generation = await client.generate(prompt, size, config.quality)
        except ImageGenerationError as error:
            raise ToolError(str(error)) from error

        mime = sniff(generation.data)
        if mime is None or not mime.startswith('image/'):
            raise ToolError('the vendor answered with something that is not an image')

        return GeneratedImage(
            data=generation.data,
            mime=mime,
            revised_prompt=generation.revised_prompt,
        )
