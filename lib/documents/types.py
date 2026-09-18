from dataclasses import dataclass


@dataclass(frozen=True)
class Extraction:
    text: str
    pages: int | None = None
    sheets: tuple[str, ...] = ()
