import csv
import io
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

type DocumentFormat = Literal['csv', 'docx', 'md', 'xlsx']
type Cell = str | int | float | bool | None

_MIMES: dict[DocumentFormat, str] = {
    'csv': 'text/csv',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'md': 'text/markdown',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
}
_FALLBACK_NAME = 'document'
_NAME_LIMIT = 100
_UNSAFE_NAME = re.compile(r'[\x00-\x1f\x7f"<>:|?*]+')
_HEADING = re.compile(r'^(#{1,6})\s+(.*)$')
_BULLET = re.compile(r'^\s*[-*+]\s+(.*)$')
_NUMBERED = re.compile(r'^\s*\d+[.)]\s+(.*)$')
_TABLE_RULE = re.compile(r'^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$')


@dataclass(frozen=True)
class Sheet:
    name: str
    rows: tuple[tuple[Cell, ...], ...]


@dataclass(frozen=True)
class Rendered:
    data: bytes
    mime: str
    name: str


class RenderError(Exception):
    pass


# NOTE: the name reaches a header and a telegram upload; only its last segment
# survives, and the extension is the one the format really has
def safe_filename(name: str, format: DocumentFormat) -> str:
    base = re.split(r'[/\\]', name)[-1]
    stem = PurePosixPath(_UNSAFE_NAME.sub('_', base)).stem.strip(' ._')
    stem = stem[: _NAME_LIMIT - len(format) - 1] or _FALLBACK_NAME
    return f'{stem}.{format}'


def render(
    format: DocumentFormat,
    filename: str,
    content: str,
    sheets: Sequence[Sheet],
) -> Rendered:
    match format:
        case 'md':
            data = _required(content, format).encode()
        case 'csv':
            data = _csv(content, sheets)
        case 'xlsx':
            data = _office(lambda: _xlsx(sheets))
        case 'docx':
            data = _office(lambda: _docx(_required(content, format)))

    return Rendered(
        data=data,
        mime=_MIMES[format],
        name=safe_filename(filename, format),
    )


def _required(content: str, format: DocumentFormat) -> str:
    if not content.strip():
        raise RenderError(f'a {format} document needs content')

    return content


def _office(write: Callable[[], bytes]) -> bytes:
    try:
        return write()
    except ImportError as error:
        raise RenderError(
            'office files need the documents extra; install huginn[documents]',
        ) from error


def _xlsx(sheets: Sequence[Sheet]) -> bytes:
    if not sheets:
        raise RenderError('an xlsx document needs at least one sheet of rows')

    from .office import write_xlsx

    return write_xlsx(sheets)


def _docx(markdown: str) -> bytes:
    from .office import write_docx

    return write_docx(tuple(parse_markdown(markdown)))


def _csv(content: str, sheets: Sequence[Sheet]) -> bytes:
    if not sheets:
        return _required(content, 'csv').encode()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(
        ['' if cell is None else cell for cell in row] for row in sheets[0].rows
    )
    return buffer.getvalue().encode()


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True)
class Item:
    text: str
    ordered: bool


@dataclass(frozen=True)
class Table:
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Code:
    text: str


@dataclass(frozen=True)
class Text:
    text: str


type Block = Heading | Item | Table | Code | Text


def _cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip('|').split('|'))


def _is_row(line: str) -> bool:
    return line.strip().startswith('|') and line.strip().endswith('|')


def _opens_table(lines: Sequence[str], index: int) -> bool:
    return (
        _is_row(lines[index])
        and index + 1 < len(lines)
        and _TABLE_RULE.match(lines[index + 1]) is not None
    )


def _prose(lines: Sequence[str], index: int) -> bool:
    line = lines[index]
    return bool(
        line.strip()
        and not line.lstrip().startswith('```')
        and not _HEADING.match(line)
        and not _BULLET.match(line)
        and not _NUMBERED.match(line)
        and not _opens_table(lines, index),
    )


def parse_markdown(markdown: str) -> Iterator[Block]:
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if _prose(lines, index):
            start = index
            while index < len(lines) and _prose(lines, index):
                index += 1
            yield Text(' '.join(text.strip() for text in lines[start:index]))
        elif not line.strip():
            index += 1
        elif line.lstrip().startswith('```'):
            end = next(
                (
                    position
                    for position in range(index + 1, len(lines))
                    if lines[position].lstrip().startswith('```')
                ),
                len(lines),
            )
            yield Code('\n'.join(lines[index + 1 : end]))
            index = end + 1
        elif heading := _HEADING.match(line):
            yield Heading(len(heading.group(1)), heading.group(2).strip())
            index += 1
        elif _opens_table(lines, index):
            rows = [_cells(line)]
            index += 2
            while index < len(lines) and _is_row(lines[index]):
                rows.append(_cells(lines[index]))
                index += 1
            yield Table(tuple(rows))
        else:
            bullet = _BULLET.match(line)
            item = bullet or _NUMBERED.match(line)
            yield Item(
                '' if item is None else item.group(1).strip(),
                ordered=bullet is None,
            )
            index += 1
