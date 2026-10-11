from base64 import b64decode
from binascii import Error as Base64Error
from dataclasses import dataclass
from hashlib import sha256
from uuid import UUID

from app.contracts.auth import ApiClient
from app.contracts.channels import (
    ATTACHMENT_LIMITS,
    AttachmentKind,
    InboundAttachment,
)
from app.core.exceptions import AppError
from app.features.agents.entities import PromptAttachment
from app.features.blobs.contracts import BlobServiceContract
from app.features.channels.contracts import AttachmentReadingServiceContract
from lib.documents import SUPPORTED_MIMES
from lib.media import sniff

from .exceptions import (
    AttachmentTooLargeError,
    BlobNotFoundError,
    InvalidAttachmentError,
    UnsupportedAttachmentError,
)
from .schemas import AttachmentInput

type AttachmentError = (
    AttachmentTooLargeError
    | BlobNotFoundError
    | InvalidAttachmentError
    | UnsupportedAttachmentError
)

INLINE_LIMIT = 1024 * 1024
_IMAGES = frozenset({'image/gif', 'image/jpeg', 'image/png', 'image/webp'})
_INLINE = 'inline'
_REMOTE = (
    'remote files are not fetched; upload it with POST /v1/blobs and send'
    ' {"blob": "<id>"} instead'
)


@dataclass(frozen=True)
class Received:
    kind: AttachmentKind
    mime: str
    data: bytes


# NOTE: the bytes say what a file is, never the label the client put on it -- that
# label decides which reader gets the bytes
def classify(
    data: bytes,
    name: str | None,
) -> Received | AttachmentTooLargeError | UnsupportedAttachmentError:
    mime = sniff(data, name)
    if mime in _IMAGES:
        kind = 'image'
    elif mime is not None and mime.startswith('audio/'):
        kind = 'audio'
    elif mime in SUPPORTED_MIMES:
        kind = 'document'
    else:
        return UnsupportedAttachmentError(
            f'cannot read {name or "this file"} ({mime or "unknown type"})',
        )

    if len(data) > ATTACHMENT_LIMITS[kind]:
        return AttachmentTooLargeError(
            f'{kind} attachments are limited to {ATTACHMENT_LIMITS[kind]} bytes',
        )

    return Received(kind=kind, mime=mime, data=data)


class AttachmentIntake:
    blobs: BlobServiceContract
    reader: AttachmentReadingServiceContract

    def __init__(
        self,
        blobs: BlobServiceContract,
        reader: AttachmentReadingServiceContract,
    ) -> None:
        self.blobs = blobs
        self.reader = reader

    async def take(
        self,
        client: ApiClient,
        inputs: tuple[AttachmentInput, ...],
    ) -> tuple[PromptAttachment, ...] | AttachmentError:
        taken = []
        for input_ in inputs:
            received = await self._receive(client, input_)
            if isinstance(received, AppError):
                return received

            attachment, data = received
            if (read := await self.reader.read(attachment, data)) is not None:
                taken.append(read)

        return tuple(taken)

    async def _receive(
        self,
        client: ApiClient,
        input_: AttachmentInput,
    ) -> tuple[InboundAttachment, bytes] | AttachmentError:
        if input_.blob is not None:
            return await self._stored(client, input_.blob)
        if input_.data is not None:
            return self._inline(input_.data, input_.name)
        if input_.url is not None:
            return InvalidAttachmentError(_REMOTE)

        return InvalidAttachmentError('an attachment needs a blob or its data')

    async def _stored(
        self,
        client: ApiClient,
        blob: str,
    ) -> tuple[InboundAttachment, bytes] | AttachmentError:
        missing = BlobNotFoundError(f'no blob {blob!r} for this key')
        try:
            id_ = UUID(blob)
        except ValueError:
            return missing

        grant = await self.blobs.granted(id_, client.key)
        if grant is None:
            return missing

        found = await self.blobs.load(id_)
        if found is None or found.data is None:
            return missing

        received = classify(found.data, grant.filename)
        if not isinstance(received, Received):
            return received

        return (
            _inbound(received, str(found.id), found.sha256, grant.filename),
            found.data,
        )

    def _inline(
        self,
        encoded: str,
        name: str | None,
    ) -> tuple[InboundAttachment, bytes] | AttachmentError:
        # NOTE: checked on the encoded length first, so an oversized body is never
        # decoded just to be refused
        if len(encoded) > INLINE_LIMIT * 4 // 3 + 4:
            return AttachmentTooLargeError(
                f'inline attachments are limited to {INLINE_LIMIT} bytes;'
                ' upload larger files with POST /v1/blobs',
            )

        try:
            data = b64decode(encoded, validate=True)
        except Base64Error, ValueError:
            return InvalidAttachmentError('attachment data is not valid base64')

        if not data:
            return InvalidAttachmentError('attachment data is empty')

        received = classify(data, name)
        if not isinstance(received, Received):
            return received

        digest = sha256(data).hexdigest()
        return _inbound(received, _INLINE, digest, name), data


def _inbound(
    received: Received,
    reference: str,
    unique: str,
    name: str | None,
) -> InboundAttachment:
    return InboundAttachment(
        kind=received.kind,
        mime=received.mime,
        reference=reference,
        unique=unique,
        name=name,
        size=len(received.data),
    )
