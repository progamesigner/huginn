import csv
import io
import json

from .exceptions import DocumentExtractionError
from .tables import markdown_table
from .types import Extraction

_PDF = 'application/pdf'
_XLSX = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
_DOCX = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
_TEXT = frozenset({'text/plain', 'text/markdown', 'text/csv', 'application/json'})

SUPPORTED_MIMES = _TEXT | {_PDF, _XLSX, _DOCX}


def extract(mime: str, data: bytes, filename: str | None = None) -> Extraction:
    name = filename or 'document'
    if mime not in SUPPORTED_MIMES:
        raise DocumentExtractionError(f'unsupported document type {mime}')

    if mime in _TEXT:
        try:
            return _text(mime, data.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DocumentExtractionError(f'could not decode {name}') from error

    try:
        return _binary(mime, data)
    except DocumentExtractionError:
        raise
    except ImportError as error:
        raise DocumentExtractionError(
            f'{mime} needs the documents extra; install huginn[documents]',
        ) from error
    except Exception as error:
        # NOTE: each parser raises its own family for a corrupt file, so the boundary
        # is the only place that can name them all
        raise DocumentExtractionError(f'could not extract {name}') from error


def _text(mime: str, text: str) -> Extraction:
    match mime:
        case 'text/csv':
            return Extraction(markdown_table(csv.reader(io.StringIO(text))))
        case 'application/json':
            value = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            return Extraction(f'```json\n{value}\n```')
        case _:
            return Extraction(text)


def _binary(mime: str, data: bytes) -> Extraction:
    match mime:
        case 'application/pdf':
            from .adapters.pdf import read_pdf

            return read_pdf(data)
        case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            from .adapters.xlsx import read_xlsx

            return read_xlsx(data)
        case _:
            from .adapters.docx import read_docx

            return read_docx(data)
