from datetime import timedelta
from typing import Annotated

from pydantic import BeforeValidator, Field

from app.core.config.registry import SectionSpec
from app.core.config.schema import ConfigValue
from lib.temporal import parse_duration


def _duration(value: object) -> object:
    return parse_duration(value) if isinstance(value, str) else value


type Duration = Annotated[timedelta | None, BeforeValidator(_duration)]


class RetentionConfig(ConfigValue):
    generated: Annotated[timedelta, BeforeValidator(_duration), Field()] = timedelta(
        days=30,
    )
    inbound: Annotated[Duration, Field()] = None


class BlobsConfig(ConfigValue):
    retention: Annotated[RetentionConfig, Field()] = RetentionConfig()


BLOBS = SectionSpec('blobs', BlobsConfig)
