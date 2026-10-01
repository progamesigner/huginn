import json
from collections.abc import Mapping
from typing import override

import yaml
from pydantic import JsonValue

from app.features.skills.entities import Skill, SkillMeta
from app.features.skills.sources.contracts import SkillSource
from lib.mcp import MCPClientContract
from lib.observability import Logger


def _split(content: str) -> tuple[Mapping[str, object], str]:
    if content.startswith('---'):
        _, _, rest = content.partition('---')
        front, sep, body = rest.partition('---')
        if sep:
            parsed = yaml.safe_load(front)
            if isinstance(parsed, dict):
                return parsed, body.lstrip('\n')
    return {}, content


def _name_from_path(path: str) -> str:
    parts = [part for part in path.split('/') if part]
    return parts[-2] if len(parts) >= 2 else path


def _meta(path: str, content: str, *, logger: Logger) -> SkillMeta:
    front, _ = _split(content)
    name = str(front.get('name') or _name_from_path(path))
    description = front.get('description')
    if not description:
        logger.warning('skill %s has no description; using its name', path)
        description = name
    return SkillMeta(name=name, description=str(description), path=path)


class MuninnSkillSource(SkillSource):
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
    ) -> None:
        self.logger = logger.bind(component='skills')
        self.client = client
        self.server = server
        self.glob = glob

    @override
    async def list(
        self,
        scope: Mapping[str, JsonValue],
    ) -> tuple[SkillMeta, ...]:
        raw = await self.client.call_tool(
            self.server,
            'list_memory_notes',
            {'glob': self.glob, **scope},
        )
        metas = [
            _meta(path, await self._read(path, scope), logger=self.logger)
            for path in _paths(raw)
        ]
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
            'read_memory_note',
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
