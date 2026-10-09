import asyncio
from dataclasses import dataclass

from app.contracts.channels import InboundAttachment
from lib.documents import DocumentExtractionError, extract
from lib.observability import Logger

_PREVIEW = 2000


def attachment_hint(attachment: InboundAttachment) -> str:
    if attachment.width is not None and attachment.height is not None:
        detail = f' {attachment.width}x{attachment.height}'
    elif attachment.duration is not None:
        detail = f' {attachment.duration}s'
    else:
        detail = ''

    if attachment.name is not None:
        detail = f' {attachment.name}{detail}'

    return f'[{attachment.kind}: {attachment.mime}{detail}]'


@dataclass(frozen=True)
class DocumentReading:
    transcript: str
    text: str | None = None


async def read_document(
    mime: str,
    data: bytes,
    name: str | None,
    hint: str,
    *,
    logger: Logger,
) -> DocumentReading:
    try:
        extracted = await asyncio.to_thread(extract, mime, data, name)
    except DocumentExtractionError as error:
        logger.warning('Could not extract document %s: %s', name or mime, error)
        return DocumentReading(f'{hint} ({error})')

    stats = []
    if extracted.pages is not None:
        stats.append(f'{extracted.pages} pages')
    if extracted.sheets:
        stats.append(f'sheets: {", ".join(extracted.sheets)}')
    if len(extracted.text) > _PREVIEW:
        stats.append('truncated')
    suffix = f' ({"; ".join(stats)})' if stats else ''

    return DocumentReading(
        f'{hint}{suffix}\n{extracted.text[:_PREVIEW]}',
        extracted.text,
    )
