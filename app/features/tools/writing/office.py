import io
import re
from collections.abc import Sequence

from docx import Document
from docx.document import Document as DocxDocument
from docx.shared import Pt
from docx.text.paragraph import Paragraph
from openpyxl import Workbook

from .renderers import Block, Code, Heading, Item, Sheet, Table, Text

_SHEET_UNSAFE = re.compile(r'[\[\]:*?/\\]+')
_SHEET_LIMIT = 31
_INLINE = re.compile(r'(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|_[^_]+_)')
_MONOSPACE = 'Courier New'


def _sheet_title(name: str, index: int, used: set[str]) -> str:
    title = _SHEET_UNSAFE.sub('_', name).strip("' ")[:_SHEET_LIMIT] or f'Sheet{index}'
    while title.lower() in used:
        title = f'{title[: _SHEET_LIMIT - len(str(index)) - 1]}_{index}'
        index += 1
    used.add(title.lower())
    return title


def write_xlsx(sheets: Sequence[Sheet]) -> bytes:
    workbook = Workbook()
    if (default := workbook.active) is not None:
        workbook.remove(default)
    used = set()
    for index, sheet in enumerate(sheets, start=1):
        worksheet = workbook.create_sheet(_sheet_title(sheet.name, index, used))
        for row_index, row in enumerate(sheet.rows, start=1):
            for column_index, value in enumerate(row, start=1):
                cell = worksheet.cell(row=row_index, column=column_index, value=value)
                # NOTE: text that looks like a formula stays text -- a model-written
                # cell must never become something excel evaluates
                if isinstance(value, str) and value.startswith('='):
                    cell.data_type = 's'

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _runs(paragraph: Paragraph, text: str) -> None:
    for piece in _INLINE.split(text):
        if not piece:
            continue
        if piece.startswith('**') and piece.endswith('**') and len(piece) > 4:
            paragraph.add_run(piece[2:-2]).bold = True
        elif piece.startswith('`') and piece.endswith('`') and len(piece) > 2:
            paragraph.add_run(piece[1:-1]).font.name = _MONOSPACE
        elif piece[0] in '*_' and piece[-1] == piece[0] and len(piece) > 2:
            paragraph.add_run(piece[1:-1]).italic = True
        else:
            paragraph.add_run(piece)


def _write(document: DocxDocument, block: Block) -> None:
    match block:
        case Heading(level=level, text=text):
            _runs(document.add_heading(level=level), text)
        case Item(text=text, ordered=ordered):
            _runs(
                document.add_paragraph(
                    style='List Number' if ordered else 'List Bullet',
                ),
                text,
            )
        case Table(rows=rows):
            width = max(len(row) for row in rows)
            table = document.add_table(rows=len(rows), cols=width)
            table.style = 'Table Grid'
            for row_index, row in enumerate(rows):
                for column_index, value in enumerate(row):
                    cell = table.cell(row_index, column_index).paragraphs[0]
                    _runs(cell, value)
                    if row_index == 0:
                        for run in cell.runs:
                            run.bold = True
        case Code(text=text):
            run = document.add_paragraph(style='No Spacing').add_run(text)
            run.font.name = _MONOSPACE
            run.font.size = Pt(9)
        case Text(text=text):
            _runs(document.add_paragraph(), text)


def write_docx(blocks: Sequence[Block]) -> bytes:
    document = Document()
    for block in blocks:
        _write(document, block)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
