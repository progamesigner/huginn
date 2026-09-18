from typing import Annotated, Literal

from pydantic import Field

from app.features.tools.config import ToolConfig


class DocumentToolConfig(ToolConfig):
    kind: Annotated[Literal['read_document'], Field()]
    max_chars: Annotated[int, Field(gt=0, le=20_000)] = 8_000
