from typing import override

from .contracts import TranscriptionServiceContract
from .repositories import TranscriptionRepository


class TranscriptionService(TranscriptionServiceContract):
    repository: TranscriptionRepository

    def __init__(self, repository: TranscriptionRepository) -> None:
        self.repository = repository

    @override
    async def transcribe(self, mime: str, data: bytes) -> str:
        return await self.repository.transcribe(mime, data)
