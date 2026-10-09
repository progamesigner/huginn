from app.exts.transcription.clients import (
    ElevenLabsTranscriptionClient,
    GeminiTranscriptionClient,
    TranscriptionClient,
    TranscriptionClientContract,
)
from lib.http import HTTPClientContract

from .config import TranscriptionConfig
from .exceptions import TranscriptionUnavailableError


class TranscriptionRepository:
    config: TranscriptionConfig
    http: HTTPClientContract

    def __init__(
        self,
        http: HTTPClientContract,
        config: TranscriptionConfig,
    ) -> None:
        self.config = config
        self.http = http

    async def transcribe(self, mime: str, data: bytes) -> str:
        return await self._client().transcribe(data, mime)

    def _client(self) -> TranscriptionClientContract:
        config = self.config
        api_key, model = config.api_key, config.model
        if config.kind is None or api_key is None or model is None:
            raise TranscriptionUnavailableError('no transcription is configured')

        base_url = None if config.base_url is None else str(config.base_url)
        match config.kind:
            case 'openai':
                return TranscriptionClient(
                    http=self.http,
                    api_key=api_key,
                    base_url=base_url or TranscriptionClient.default_base_url,
                    model=model,
                )
            case 'gemini':
                return GeminiTranscriptionClient(
                    http=self.http,
                    api_key=api_key,
                    base_url=base_url or GeminiTranscriptionClient.default_base_url,
                    model=model,
                    mode=config.mode,
                    languages=config.language_codes,
                )
            case 'elevenlabs':
                return ElevenLabsTranscriptionClient(
                    http=self.http,
                    api_key=api_key,
                    base_url=base_url or ElevenLabsTranscriptionClient.default_base_url,
                    model=model,
                    language=next(iter(config.language_codes), None),
                )
