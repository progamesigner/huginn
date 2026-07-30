from typing import override

from app.exts.transcription.clients import TranscriptionClient
from lib.http import HTTPClientContract

from .config import TranscriptionConfig
from .contracts import TranscriptionContract
from .exceptions import TranscriptionUnavailableError


class TranscriptionRepository(TranscriptionContract):
    config: TranscriptionConfig
    http: HTTPClientContract

    def __init__(
        self,
        http: HTTPClientContract,
        config: TranscriptionConfig,
    ) -> None:
        self.config = config
        self.http = http

    @override
    async def transcribe(self, mime: str, data: bytes) -> str:
        api_key, model = self.config.api_key, self.config.model
        if self.config.kind is None or api_key is None or model is None:
            raise TranscriptionUnavailableError('no transcription is configured')

        client = TranscriptionClient(
            http=self.http,
            api_key=api_key,
            base_url=str(self.config.base_url),
            model=model,
        )

        return await client.transcribe(data, mime)
