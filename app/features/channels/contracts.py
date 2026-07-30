from abc import ABC, abstractmethod


class TranscriptionContract(ABC):
    @abstractmethod
    async def transcribe(self, mime: str, data: bytes) -> str: ...
