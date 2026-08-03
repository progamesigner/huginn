from typing import Annotated

from pydantic import Field

from app.core.config.registry import SectionSpec
from app.core.config.schema import ConfigValue


class CompactionConfig(ConfigValue):
    summary_prompt: Annotated[str | None, Field()] = None


COMPACTION = SectionSpec('compaction', CompactionConfig)
