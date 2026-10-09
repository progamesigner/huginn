from typing import override

from app.contracts.channels import InboundAttachment
from app.features.agents.entities import PromptAttachment
from lib.observability import Logger

from .contracts import AttachmentReadingServiceContract, TranscriptionServiceContract
from .documents import attachment_hint, read_document
from .exceptions import TranscriptionUnavailableError
from .repositories import TranscriptionRepository


class TranscriptionService(TranscriptionServiceContract):
    repository: TranscriptionRepository

    def __init__(self, repository: TranscriptionRepository) -> None:
        self.repository = repository

    @override
    async def transcribe(self, mime: str, data: bytes) -> str:
        return await self.repository.transcribe(mime, data)


class AttachmentReader(AttachmentReadingServiceContract):
    logger: Logger
    transcription: TranscriptionServiceContract

    def __init__(
        self,
        transcription: TranscriptionServiceContract,
        *,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='attachments')
        self.transcription = transcription

    @override
    async def read(
        self,
        attachment: InboundAttachment,
        data: bytes,
    ) -> PromptAttachment | None:
        hint = attachment_hint(attachment)
        transcript = None
        document_text = None
        if attachment.kind == 'audio':
            transcript = await self._transcript(attachment, data)
            if transcript is None:
                return None
        elif attachment.kind == 'document':
            reading = await read_document(
                attachment.mime,
                data,
                attachment.name,
                hint,
                logger=self.logger,
            )
            transcript = reading.transcript
            document_text = reading.text

        return PromptAttachment(
            kind=attachment.kind,
            mime=attachment.mime,
            reference=attachment.reference,
            unique=attachment.unique,
            data=data,
            hint=hint,
            name=attachment.name,
            transcript=transcript,
            document_text=document_text,
        )

    async def _transcript(
        self,
        attachment: InboundAttachment,
        data: bytes,
    ) -> str | None:
        try:
            return await self.transcription.transcribe(attachment.mime, data)
        except TranscriptionUnavailableError:
            self.logger.warning(
                'Dropping the voice of %s: no transcription is configured',
                attachment.reference,
            )
        except Exception:
            self.logger.exception(
                'Could not transcribe the voice of %s',
                attachment.reference,
            )

        return None
