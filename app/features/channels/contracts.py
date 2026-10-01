from abc import ABC, abstractmethod


class TranscriptionServiceContract(ABC):
    @abstractmethod
    async def transcribe(self, mime: str, data: bytes) -> str: ...
