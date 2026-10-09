from app.exts.speech.clients import SpeechClient
from app.exts.speech.exceptions import SpeechSynthesisError
from app.features.tools.exceptions import ToolError
from lib.http import HTTPClientContract

from .config import SpeechToolConfig


class SpeechRepository:
    http: HTTPClientContract

    def __init__(self, http: HTTPClientContract) -> None:
        self.http = http

    async def synthesize(
        self,
        config: SpeechToolConfig,
        text: str,
        voice: str,
    ) -> bytes:
        client = SpeechClient(
            http=self.http,
            api_key=config.api_key,
            base_url=str(config.base_url),
            model=config.model,
        )
        try:
            data = await client.synthesize(text, voice, config.format)
        except SpeechSynthesisError as error:
            raise ToolError(str(error)) from error

        if not data:
            raise ToolError('the vendor answered with no audio')

        return data
