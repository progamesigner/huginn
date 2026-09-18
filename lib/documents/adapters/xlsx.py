import io

from openpyxl import load_workbook

from lib.documents.tables import markdown_table
from lib.documents.types import Extraction


def read_xlsx(data: bytes) -> Extraction:
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    sheets = tuple(workbook.sheetnames)
    sections = []
    for sheet in sheets:
        worksheet = workbook[sheet]
        sections.append(
            f'## Sheet: {sheet} ({worksheet.max_row}x{worksheet.max_column})',
        )
        sections.append(markdown_table(worksheet.iter_rows(values_only=True)))

    return Extraction('\n\n'.join(sections), sheets=sheets)
