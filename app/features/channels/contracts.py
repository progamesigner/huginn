from abc import ABC, abstractmethod

from app.contracts.channels import InboundAttachment
from app.features.agents.entities import PromptAttachment


class TranscriptionServiceContract(ABC):
    @abstractmethod
    async def transcribe(self, mime: str, data: bytes) -> str: ...


# NOTE: what a channel hands the model for a file it already holds -- a voice note
# heard, a document read -- or None when there is nothing the model could use
class AttachmentReadingServiceContract(ABC):
    @abstractmethod
    async def read(
        self,
        attachment: InboundAttachment,
        data: bytes,
    ) -> PromptAttachment | None: ...
