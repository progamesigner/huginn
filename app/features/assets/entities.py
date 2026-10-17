from typing import Annotated, Literal

from pydantic import Field, JsonValue

from app.core.schema import AppValue

type AssetKind = Literal['skill', 'command', 'subagent']


class AssetMeta(AppValue):
    kind: Annotated[AssetKind, Field()]
    name: Annotated[str, Field()]
    path: Annotated[str, Field()]
    source: Annotated[str, Field()]
    frontmatter: Annotated[dict[str, JsonValue], Field()]


class Asset(AppValue):
    meta: Annotated[AssetMeta, Field()]
    body: Annotated[str, Field()]


class AssetDocument(AppValue):
    content: Annotated[str, Field()]
    version: Annotated[str | None, Field()] = None
