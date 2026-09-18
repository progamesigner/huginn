import asyncio
import logging
from dataclasses import dataclass

from lib.documents import DocumentExtractionError, extract

logger = logging.getLogger(__name__)

_PREVIEW = 2000


@dataclass(frozen=True)
class DocumentReading:
    transcript: str
    text: str | None = None


async def read_document(
    mime: str,
    data: bytes,
    name: str | None,
    hint: str,
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
