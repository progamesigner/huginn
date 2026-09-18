import io

from pypdf import PdfReader

from lib.documents.exceptions import DocumentExtractionError
from lib.documents.types import Extraction


def read_pdf(data: bytes) -> Extraction:
    pages = [page.extract_text() or '' for page in PdfReader(io.BytesIO(data)).pages]
    if not any(text.strip() for text in pages):
        raise DocumentExtractionError('PDF has no text layer')

    return Extraction(
        '\n\n'.join(
            text if text.strip() else f'[page {number}: no text layer]'
            for number, text in enumerate(pages, start=1)
        ),
        pages=len(pages),
    )
