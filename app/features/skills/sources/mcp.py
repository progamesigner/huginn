import json
from collections.abc import Mapping
from typing import override

from pydantic import JsonValue

from app.features.skills.entities import Skill, SkillMeta
from app.features.skills.sources.contracts import SkillSource
from lib.mcp import MCPClientContract
from lib.observability import Logger

from .markdown import _meta, _split


class MCPSkillSource(SkillSource):
    logger: Logger
    client: MCPClientContract
    server: str
    glob: str

    def __init__(
        self,
        client: MCPClientContract,
        server: str,
        glob: str,
        *,
        logger: Logger,
        list_tool: str = 'list_memory_notes',
        read_tool: str = 'read_memory_note',
    ) -> None:
        self.logger = logger.bind(component='skills')
        self.client = client
        self.server = server
        self.glob = glob
        self.list_tool = list_tool
        self.read_tool = read_tool

    @override
    async def list(
        self,
        scope: Mapping[str, JsonValue],
    ) -> tuple[SkillMeta, ...]:
        raw = await self.client.call_tool(
            self.server,
            self.list_tool,
            {'glob': self.glob, **scope},
        )
        metas = []
        for path in _paths(raw):
            content = await self._read(path, scope)
            try:
                metas.append(_meta(path, content, logger=self.logger))
            except TypeError, ValueError:
                self.logger.warning('Skipping invalid skill frontmatter: %s', path)
        return tuple(metas)

    @override
    async def load(self, path: str, scope: Mapping[str, JsonValue]) -> Skill:
        content = await self._read(path, scope)
        return Skill(
            meta=_meta(path, content, logger=self.logger),
            body=_split(content)[1],
        )

    async def _read(self, path: str, scope: Mapping[str, JsonValue]) -> str:
        raw = await self.client.call_tool(
            self.server,
            self.read_tool,
            {'path': path, **scope},
        )
        return _content(raw)


def _paths(raw: str) -> list[str]:
    data = json.loads(raw)
    items = data.get('items') if isinstance(data, dict) else data
    return [str(item) for item in items] if isinstance(items, list) else []


def _content(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(data, dict):
        return str(data.get('content', ''))
    return data if isinstance(data, str) else raw
