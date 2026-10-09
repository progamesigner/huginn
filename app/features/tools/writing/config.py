from typing import Annotated, Literal

from pydantic import Field

from app.features.tools.config import ToolConfig


class WritingToolConfig(ToolConfig):
    kind: Annotated[Literal['write_document'], Field()]
    max_bytes: Annotated[int, Field(gt=0, le=50 * 1024 * 1024)] = 20 * 1024 * 1024
