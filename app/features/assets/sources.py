import json
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import cast

import yaml
from pydantic import JsonValue, TypeAdapter

from lib.mcp import MCPClientContract, MCPError, MCPToolError
from lib.observability import Logger

from .config import AssetSourceConfig, FilesystemAssetSourceConfig
from .entities import Asset, AssetDocument, AssetKind, AssetMeta
from .exceptions import AssetError

_NAME = re.compile(r'^[^\s/\\]+$')
_JSON = TypeAdapter(JsonValue)


def _scalars(value: object) -> object:
    # YAML resolves unquoted timestamps to dates; metadata stays JSON-compatible.
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('frontmatter keys must be strings')
        return {key: _scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scalars(item) for item in value]
    return value


def parse(source: str, kind: AssetKind, path: str, content: str) -> Asset:
    lines = content.splitlines(keepends=True)
    front = dict[str, JsonValue]()
    body = content
    if lines and lines[0].strip() == '---':
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == '---'), None)
        if end is None:
            raise ValueError('unterminated frontmatter')
        try:
            parsed = _JSON.validate_python(
                _scalars(yaml.safe_load(''.join(lines[1:end])) or {}),
            )
        except (ValueError, yaml.YAMLError) as error:
            raise ValueError('invalid frontmatter') from error
        if not isinstance(parsed, dict):
            raise ValueError('frontmatter must be a mapping')
        front = parsed
        body = ''.join(lines[end + 1 :]).lstrip('\r\n')
    inferred = Path(path).stem if kind == 'command' else Path(path).parent.name
    name = front.get('name') or inferred
    if kind == 'skill':
        name = str(name)
    if (
        not isinstance(name, str)
        or not name.strip()
        or (kind != 'skill' and (not _NAME.fullmatch(name) or name in {'.', '..'}))
    ):
        raise ValueError('invalid asset name')
    return Asset(
        meta=AssetMeta(
            kind=kind,
            name=name,
            path=path,
            source=source,
            frontmatter=front,
        ),
        body=body,
    )


def _payload(raw: str) -> object:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    # Accept both a direct service result and MCP text/structured envelopes.
    if isinstance(data, dict) and isinstance(data.get('structuredContent'), dict):
        data = data['structuredContent']
    if isinstance(data, dict) and data.get('isError'):
        raise AssetError('asset service reported an error')
    if isinstance(data, dict) and isinstance(data.get('content'), list):
        blocks = data['content']
        text = '\n'.join(
            item['text']
            for item in blocks
            if isinstance(item, dict) and isinstance(item.get('text'), str)
        )
        return _payload(text)
    return data


class AssetSource:
    def __init__(
        self,
        name: str,
        config: AssetSourceConfig,
        client: MCPClientContract | None,
        *,
        logger: Logger,
    ) -> None:
        self.name = name
        self.config = config
        self.client = client
        self.logger = logger.bind(component='assets', source=name)

    def _client(self) -> MCPClientContract:
        if self.client is None:
            raise AssetError('MCP asset source has no client')
        return self.client

    def _path(self, path: str) -> Path:
        if not isinstance(self.config, FilesystemAssetSourceConfig):
            raise TypeError('not a filesystem source')
        root = self.config.root.resolve(strict=True)
        candidate = (root / path).resolve(strict=True)
        if not candidate.is_relative_to(root):
            raise ValueError('asset path escapes its root')
        return candidate

    async def read(
        self,
        path: str,
        scope: Mapping[str, JsonValue],
    ) -> AssetDocument | None:
        if isinstance(self.config, FilesystemAssetSourceConfig):
            try:
                return AssetDocument(
                    content=self._path(path).read_text(encoding='utf-8'),
                )
            except FileNotFoundError:
                return None
        arguments = {**scope, 'path': path}
        if path == 'Agents/AGENTS.md':
            # Older memory services can still supply a read-only brief. Request
            # versions only when the read schema advertises the opt-in field.
            try:
                infos = await self._client().list_tools(self.config.server)
            except MCPError:
                infos = ()
            for info in infos:
                if info.name != self.config.read_tool or not isinstance(
                    info.input_schema,
                    dict,
                ):
                    continue
                properties = info.input_schema.get('properties')
                if isinstance(properties, dict) and 'include_version' in properties:
                    arguments['include_version'] = True
        read = (
            self._client().call_tool_result
            if arguments.get('include_version')
            else self._client().call_tool
        )
        try:
            raw = _payload(
                await read(
                    self.config.server,
                    self.config.read_tool,
                    arguments,
                ),
            )
        except MCPToolError as error:
            if error.code == 'not_found':
                return None
            raise
        if isinstance(raw, dict):
            if raw.get('code') == 'not_found':
                return None
            if raw.get('code') or raw.get('error'):
                raise AssetError('asset read failed')
            content = raw.get('content')
            if not isinstance(content, str):
                raise AssetError('asset read returned no content')
            version = raw.get('version')
            return AssetDocument(
                content=content,
                version=version if isinstance(version, str) else None,
            )
        if isinstance(raw, str):
            return AssetDocument(content=raw)
        raise AssetError('invalid asset read response')

    async def paths(
        self,
        kind: AssetKind,
        scope: Mapping[str, JsonValue],
    ) -> tuple[str, ...]:
        glob = self.config.globs.for_kind(kind)
        if isinstance(self.config, FilesystemAssetSourceConfig):
            root = self.config.root.resolve(strict=True)
            return tuple(
                self._path(str(path)).relative_to(root).as_posix()
                for path in sorted(root.glob(glob))
                if path.is_file()
            )
        paths = list[str]()
        cursor = None
        seen = set()
        while True:
            args = {**scope, 'glob': glob}
            if cursor is not None:
                args['cursor'] = cursor
            data = _payload(
                await self._client().call_tool(
                    self.config.server,
                    self.config.list_tool,
                    args,
                ),
            )
            items = data.get('items') if isinstance(data, dict) else data
            if not isinstance(items, list) or any(
                not isinstance(path, str) for path in items
            ):
                raise AssetError('invalid asset inventory response')
            paths.extend(cast(list[str], items))
            cursor = data.get('next_cursor') if isinstance(data, dict) else None
            if cursor is None:
                return tuple(dict.fromkeys(paths))
            if not isinstance(cursor, str) or cursor in seen:
                raise AssetError('invalid asset inventory cursor')
            seen.add(cursor)

    async def inventory(
        self,
        kind: AssetKind,
        scope: Mapping[str, JsonValue],
    ) -> tuple[AssetMeta, ...]:
        metas = dict[str, AssetMeta]()
        duplicates = set[str]()
        # Sorted, so a duplicate skill always resolves to the same path.
        for path in sorted(await self.paths(kind, scope)):
            document = await self.read(path, scope)
            if document is None:
                continue
            try:
                meta = parse(self.name, kind, path, document.content).meta
            except ValueError:
                self.logger.warning('Skipping invalid %s metadata at %s', kind, path)
                continue
            if meta.name in metas and kind == 'skill':
                self.logger.warning(
                    'Skipping duplicate skill %r at %s; keeping %s',
                    meta.name,
                    path,
                    metas[meta.name].path,
                )
                continue
            if meta.name in metas:
                # A duplicated command or subagent is withheld rather than
                # guessed; the rest of the inventory stays usable.
                self.logger.warning(
                    'Withholding duplicate %s %r at %s and %s',
                    kind,
                    meta.name,
                    metas[meta.name].path,
                    path,
                )
                duplicates.add(meta.name)
                continue
            metas[meta.name] = meta
        return tuple(meta for name, meta in metas.items() if name not in duplicates)
