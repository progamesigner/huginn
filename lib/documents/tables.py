from collections.abc import Iterable


def markdown_table(rows: Iterable[Iterable[object]]) -> str:
    cells = [
        tuple('' if value is None else str(value) for value in row) for row in rows
    ]
    if not cells:
        return ''

    width = max(len(row) for row in cells)
    header, *body = (row + ('',) * (width - len(row)) for row in cells)
    return '\n'.join(
        '| ' + ' | '.join(row) + ' |' for row in (header, ('---',) * width, *body)
    )
