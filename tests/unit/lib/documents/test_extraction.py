import io
import sys

import pytest
from docx import Document
from openpyxl import Workbook

from lib.documents import SUPPORTED_MIMES, DocumentExtractionError, extract

_XLS = 'application/vnd.ms-excel'
_PPTX = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'


def _pdf(*pages: str) -> bytes:
    objects = ['<< /Type /Catalog /Pages 2 0 R >>']
    kids = ' '.join(f'{3 + index * 2} 0 R' for index in range(len(pages)))
    objects.append(f'<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>')
    font = 3 + len(pages) * 2
    for index, text in enumerate(pages):
        stream = f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET' if text else ''
        objects.append(
            f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
            f'/Resources << /Font << /F1 {font} 0 R >> >> '
            f'/Contents {4 + index * 2} 0 R >>',
        )
        objects.append(f'<< /Length {len(stream)} >>\nstream\n{stream}\nendstream')
    objects.append('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')

    body = b'%PDF-1.4\n'
    offsets = []
    for number, content in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f'{number} 0 obj\n{content}\nendobj\n'.encode()
    xref = len(body)
    body += f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode()
    body += b''.join(f'{offset:010} 00000 n \n'.encode() for offset in offsets)
    body += (
        f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n'
        f'startxref\n{xref}\n%%EOF\n'
    ).encode()
    return body


def test_extracts_plain_text() -> None:
    result = extract('text/plain', 'hello 文件'.encode())

    assert result.text == 'hello 文件'
    assert result.pages is None


def test_extracts_csv_as_markdown() -> None:
    result = extract('text/csv', b'name,total\ncoffee,2\n')

    assert result.text == '| name | total |\n| --- | --- |\n| coffee | 2 |'


def test_extracts_json_as_fenced_text() -> None:
    result = extract('application/json', b'{"answer": 42}')

    assert result.text == '```json\n{\n  "answer": 42\n}\n```'


def test_undecodable_text_names_the_file() -> None:
    with pytest.raises(DocumentExtractionError, match=r'could not decode notes\.txt'):
        extract('text/plain', b'\xff\xfe\xfa', 'notes.txt')


@pytest.mark.parametrize('mime', ['application/zip', _XLS, _PPTX])
def test_rejects_types_it_cannot_read(mime: str) -> None:
    assert mime not in SUPPORTED_MIMES
    with pytest.raises(DocumentExtractionError, match='unsupported document type'):
        extract(mime, b'bytes')


def test_a_pdf_marks_a_page_without_text_instead_of_failing() -> None:
    result = extract('application/pdf', _pdf('Cover words', '', 'Last words'))

    assert result.pages == 3
    assert 'Cover words' in result.text
    assert '[page 2: no text layer]' in result.text
    assert 'Last words' in result.text


def test_a_pdf_with_no_text_at_all_is_refused() -> None:
    with pytest.raises(DocumentExtractionError, match='no text layer'):
        extract('application/pdf', _pdf('', ''))


def test_a_corrupt_pdf_is_reported_by_name() -> None:
    with pytest.raises(DocumentExtractionError, match=r'could not extract scan\.pdf'):
        extract('application/pdf', b'not a pdf', 'scan.pdf')


def test_extracts_every_sheet_of_a_workbook() -> None:
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = 'Costs'
    sheet.append(['item', 'total'])
    sheet.append(['coffee', None])
    buffer = io.BytesIO()
    workbook.save(buffer)

    result = extract(
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        buffer.getvalue(),
    )

    assert result.sheets == ('Costs',)
    assert result.text == (
        '## Sheet: Costs (2x2)\n\n| item | total |\n| --- | --- |\n| coffee |  |'
    )


def test_extracts_word_paragraphs() -> None:
    document = Document()
    document.add_paragraph('first')
    document.add_paragraph('second')
    buffer = io.BytesIO()
    document.save(buffer)

    result = extract(
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        buffer.getvalue(),
    )

    assert result.text == 'first\n\nsecond'


def test_a_missing_parser_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'lib.documents.adapters.pdf', None)

    with pytest.raises(DocumentExtractionError, match=r'huginn\[documents\]'):
        extract('application/pdf', b'%PDF')
