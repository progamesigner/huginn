import json
import logging
from collections.abc import Mapping
from typing import override

import yaml
from pydantic import JsonValue

from app.features.skills.sources.contracts import SkillSource
from app.features.skills.types import Skill, SkillMeta
from lib.mcp import MCPClientContract

logger = logging.getLogger(__name__)


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


def _meta(path: str, content: str) -> SkillMeta:
    front, _ = _split(content)
    name = str(front.get('name') or _name_from_path(path))
    description = front.get('description')
    if not description:
        logger.warning('skill %s has no description; using its name', path)
        description = name
    return SkillMeta(name=name, description=str(description), path=path)


class MuninnSkillSource(SkillSource):
    client: MCPClientContract
    server: str
    glob: str
    scope: dict[str, JsonValue]

    def __init__(
        self,
        client: MCPClientContract,
        server: str,
        scope: Mapping[str, JsonValue],
        glob: str,
    ) -> None:
        self.client = client
        self.server = server
        self.scope = dict(scope)
        self.glob = glob

    @override
    async def list(self) -> tuple[SkillMeta, ...]:
        raw = await self.client.call_tool(
            self.server,
            'list_memory_notes',
            {'glob': self.glob, **self.scope},
        )
        metas = [_meta(path, await self._read(path)) for path in _paths(raw)]
        return tuple(metas)

    @override
    async def load(self, path: str) -> Skill:
        content = await self._read(path)
        return Skill(meta=_meta(path, content), body=_split(content)[1])

    async def _read(self, path: str) -> str:
        raw = await self.client.call_tool(
            self.server,
            'read_memory_note',
            {'path': path, **self.scope},
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
