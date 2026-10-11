import csv
import io
import sys

import pytest
from docx import Document
from openpyxl import load_workbook

from app.features.tools.writing.renderers import (
    Cell,
    Code,
    Heading,
    Item,
    RenderError,
    Sheet,
    Table,
    Text,
    parse_markdown,
    render,
    safe_filename,
)


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('report', 'report.xlsx'),
        ('report.pdf', 'report.xlsx'),
        ('../../etc/passwd', 'passwd.xlsx'),
        ('..\\..\\windows\\evil.exe', 'evil.xlsx'),
        ('quarterly\r\nX-Evil: 1', 'quarterly_X-Evil_ 1.xlsx'),
        ('', 'document.xlsx'),
        ('...', 'document.xlsx'),
        ('a' * 300, 'a' * 95 + '.xlsx'),
    ],
)
def test_a_file_name_keeps_only_a_safe_last_segment(name: str, expected: str) -> None:
    assert safe_filename(name, 'xlsx') == expected


def test_an_xlsx_keeps_numbers_as_numbers() -> None:
    rendered = render(
        'xlsx',
        'sales',
        '',
        (Sheet('Q1', (('month', 'revenue'), ('Jan', 1200), ('Feb', 1350.5))),),
    )

    workbook = load_workbook(io.BytesIO(rendered.data))
    sheet = workbook['Q1']
    assert [[cell.value for cell in row] for row in sheet.iter_rows()] == [
        ['month', 'revenue'],
        ['Jan', 1200],
        ['Feb', 1350.5],
    ]
    assert isinstance(sheet['B2'].value, int)
    assert isinstance(sheet['B3'].value, float)
    assert rendered.name == 'sales.xlsx'
    assert rendered.mime == (
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


def test_an_xlsx_never_turns_model_text_into_a_formula() -> None:
    rendered = render('xlsx', 'x', '', (Sheet('S', (('=1+1',),)),))

    cell = load_workbook(io.BytesIO(rendered.data))['S']['A1']
    assert cell.data_type == 's'
    assert cell.value == '=1+1'


def test_xlsx_sheet_names_are_cleaned_and_kept_apart() -> None:
    rendered = render(
        'xlsx',
        'x',
        '',
        (Sheet('a/b:c', ((1,),)), Sheet('A_b_c', ((2,),)), Sheet('', ((3,),))),
    )

    assert load_workbook(io.BytesIO(rendered.data)).sheetnames == [
        'a_b_c',
        'A_b_c_2',
        'Sheet3',
    ]


def test_an_xlsx_without_sheets_is_refused() -> None:
    with pytest.raises(RenderError, match='at least one sheet'):
        render('xlsx', 'x', 'a,b', ())


def test_a_csv_is_its_content_or_its_first_sheet() -> None:
    assert render('csv', 'x', 'a,b\n1,2\n', ()).data == b'a,b\n1,2\n'
    rows = render('csv', 'x', '', (Sheet('s', (('a', None), (1, 'x,y'))),)).data
    assert rows == b'a,\r\n1,"x,y"\r\n'


@pytest.mark.parametrize('from_content', [False, True])
@pytest.mark.parametrize(
    ('cell', 'expected'),
    [
        ('=1+1', "'=1+1"),
        ('+1+1', "'+1+1"),
        ('-1+1', "'-1+1"),
        ('@SUM(1,2)', "'@SUM(1,2)"),
        ('\t=1+1', "'\t=1+1"),
        ('\r=1+1', "'\r=1+1"),
        (-1, "'-1"),
        (-1.5, "'-1.5"),
        ('', ''),
        (None, ''),
        (1, '1'),
        (1.5, '1.5'),
        (True, 'True'),
        ('plain text', 'plain text'),
        ("'=1+1", "'=1+1"),
        ('a=1+1', 'a=1+1'),
        ('x,y', 'x,y'),
        ('a"b\nc', 'a"b\nc'),
    ],
)
def test_csv_cells_are_safe_for_spreadsheets(
    cell: Cell,
    expected: str,
    from_content: bool,
) -> None:
    rows = (('header', 'value'), ('row', cell), ('tail', 'unchanged'))
    if from_content:
        buffer = io.StringIO()
        csv.writer(buffer).writerows(rows)
        rendered = render('csv', 'x', buffer.getvalue(), ())
    else:
        rendered = render('csv', 'x', '', (Sheet('S', rows),))

    assert list(csv.reader(io.StringIO(rendered.data.decode(), newline=''))) == [
        ['header', 'value'],
        ['row', expected],
        ['tail', 'unchanged'],
    ]


def test_markdown_and_word_need_content() -> None:
    with pytest.raises(RenderError, match='md document needs content'):
        render('md', 'x', '  ', ())
    with pytest.raises(RenderError, match='docx document needs content'):
        render('docx', 'x', '', ())


def test_markdown_is_kept_as_written() -> None:
    rendered = render('md', 'notes', '# 標題\n\n內容', ())

    assert rendered.data == '# 標題\n\n內容'.encode()
    assert (rendered.mime, rendered.name) == ('text/markdown', 'notes.md')


@pytest.mark.parametrize(
    ('markdown', 'blocks'),
    [
        ('# Title', [Heading(1, 'Title')]),
        ('### Deep', [Heading(3, 'Deep')]),
        ('- one\n* two', [Item('one', ordered=False), Item('two', ordered=False)]),
        (
            '1. first\n2) second',
            [Item('first', ordered=True), Item('second', ordered=True)],
        ),
        (
            '| a | b |\n|---|:-:|\n| 1 | 2 |',
            [Table((('a', 'b'), ('1', '2')))],
        ),
        ('```\nx = 1\n```', [Code('x = 1')]),
        ('line one\nline two\n\nnext', [Text('line one line two'), Text('next')]),
        ('intro\n- item', [Text('intro'), Item('item', ordered=False)]),
    ],
)
def test_markdown_maps_onto_word_blocks(markdown: str, blocks: list[object]) -> None:
    assert list(parse_markdown(markdown)) == blocks


def test_a_word_document_carries_headings_lists_tables_and_emphasis() -> None:
    rendered = render(
        'docx',
        'memo',
        '# Plan\n\nSome **bold** and *soft* words.\n\n- first\n1. second\n\n'
        '| k | v |\n|---|---|\n| a | 1 |\n\n```\ncode\n```',
        (),
    )

    document = Document(io.BytesIO(rendered.data))
    styled = [
        (paragraph.style and paragraph.style.name, paragraph.text)
        for paragraph in document.paragraphs
    ]
    assert styled == [
        ('Heading 1', 'Plan'),
        ('Normal', 'Some bold and soft words.'),
        ('List Bullet', 'first'),
        ('List Number', 'second'),
        ('No Spacing', 'code'),
    ]
    runs = document.paragraphs[1].runs
    assert [(run.text, run.bold, run.italic) for run in runs] == [
        ('Some ', None, None),
        ('bold', True, None),
        (' and ', None, None),
        ('soft', None, True),
        (' words.', None, None),
    ]
    (table,) = document.tables
    assert [[cell.text for cell in row.cells] for row in table.rows] == [
        ['k', 'v'],
        ['a', '1'],
    ]


def test_office_files_without_the_documents_extra_are_refused_plainly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, 'app.features.tools.writing.office', raising=False)
    monkeypatch.setitem(sys.modules, 'openpyxl', None)

    with pytest.raises(RenderError, match='documents extra'):
        render('xlsx', 'x', '', (Sheet('S', ((1,),)),))
