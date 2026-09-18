import io

from docx import Document

from lib.documents.types import Extraction


def read_docx(data: bytes) -> Extraction:
    document = Document(io.BytesIO(data))

    return Extraction('\n\n'.join(paragraph.text for paragraph in document.paragraphs))
