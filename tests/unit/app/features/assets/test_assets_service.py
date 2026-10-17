import asyncio
import json
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import override

import pytest
from pydantic import JsonValue

from app.features.assets.config import (
    AssetGlobs,
    AssetsConfig,
    FilesystemAssetSourceConfig,
    MCPAssetSourceConfig,
)
from app.features.assets.exceptions import AssetError
from app.features.assets.services import AssetsService
from lib.mcp import MCPClientContract, MCPToolInfo
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger


class Memory(MCPClientContract):
    def __init__(self) -> None:
        self.calls = list[tuple[str, dict[str, JsonValue]]]()
        self.body = 'initial'
        self.gate: asyncio.Event | None = None

    @override
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, JsonValue],
    ) -> str:
        self.calls.append((tool, dict(arguments)))
        if self.gate is not None:
            await self.gate.wait()
        if tool == 'list_memory_notes':
            if 'PROMPT.md' in str(arguments.get('glob')):
                return json.dumps({'items': ['Agents/agents/check/PROMPT.md']})
            return json.dumps(
                {'items': ['Agents/commands/check.md'], 'next_cursor': 'p2'}
                if 'cursor' not in arguments
                else {'items': ['Agents/commands/other.md']},
            )
        return json.dumps(
            {
                'content': f'---\ndescription: works\ncustom: [1, 2]\n---\n{server}/{arguments.get("user")}/{self.body}',
                'version': 'v1',
            },
        )

    @override
    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        return ()

    @override
    def unavailable(self) -> tuple[str, ...]:
        return ()


def service(memory: Memory, clock: FakeClock, max_entries: int = 256) -> AssetsService:
    return AssetsService(
        AssetsConfig(
            sources={
                name: MCPAssetSourceConfig(server=name) for name in ('one', 'two')
            },
            max_entries=max_entries,
        ),
        memory,
        clock,
        logger=make_logger(),
    )


async def test_scopes_sources_kinds_and_fresh_body() -> None:
    memory, clock = Memory(), FakeClock()
    assets = service(memory, clock)
    for source, user in (('one', 'alice'), ('one', 'bob'), ('two', 'alice')):
        result = await assets.load(source, 'command', 'check', {'user': user})
        assert result is not None
        assert result.body == f'{source}/{user}/initial'
        assert result.meta.frontmatter['custom'] == [1, 2]
    memory.body = 'edited'
    result = await assets.load('one', 'command', 'check', {'user': 'alice'})
    assert result is not None
    assert result.body.endswith('/edited')
    assert len([call for call in memory.calls if call[0] == 'list_memory_notes']) == 6
    await assets.inventory('one', 'subagent', {'user': 'alice'})
    assert len(assets.cache) == 4
    clock.advance(timedelta(minutes=16))
    await assets.inventory('one', 'command', {'user': 'alice'})
    assert len(assets.cache) == 1
    assets.invalidate('one')
    assert not assets.cache


async def test_concurrent_refresh_is_coalesced_and_capacity_is_bounded() -> None:
    memory, clock = Memory(), FakeClock()
    memory.gate = asyncio.Event()
    assets = service(memory, clock, 1)
    first = asyncio.create_task(assets.inventory('one', 'command', {'user': 'alice'}))
    second = asyncio.create_task(assets.inventory('one', 'command', {'user': 'alice'}))
    await asyncio.sleep(0)
    memory.gate.set()
    assert await first == await second
    assert len([call for call in memory.calls if call[0] == 'list_memory_notes']) == 2
    await assets.inventory('one', 'command', {'user': 'bob'})
    assert len(assets.cache) == 1
    assert not assets.pending


async def test_filesystem_containment_metadata_and_collision(tmp_path: Path) -> None:
    root = tmp_path / 'assets'
    root.mkdir()
    (root / 'one.md').write_text('---\nname: same\n---\nhello')
    (root / 'bad.md').write_text('---\nname: [broken]\n---\nignored')
    assets = AssetsService(
        AssetsConfig(
            sources={
                'files': FilesystemAssetSourceConfig(
                    kind='filesystem',
                    root=root,
                    globs=AssetGlobs(command='*.md'),
                ),
            },
        ),
        Memory(),
        FakeClock(),
        logger=make_logger(),
    )
    assert [meta.name for meta in await assets.inventory('files', 'command', {})] == [
        'same',
    ]
    assert await assets.read_one('files', 'missing', {}) is None
    outside = tmp_path / 'secret'
    outside.write_text('secret')
    (root / 'escape').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        await assets.read_one('files', 'escape', {})
    (root / 'two.md').write_text('---\nname: same\n---\ncollision')
    (root / 'three.md').write_text('---\nname: other\n---\nfine')
    assets.invalidate()
    # The duplicated name is withheld; the rest of the inventory survives.
    assert [meta.name for meta in await assets.inventory('files', 'command', {})] == [
        'other',
    ]


async def test_read_failure_is_not_a_missing_document() -> None:
    class Broken(Memory):
        @override
        async def call_tool(
            self,
            server: str,
            tool: str,
            arguments: Mapping[str, JsonValue],
        ) -> str:
            return '{"code":"permission_denied"}'

    with pytest.raises(AssetError):
        await service(Broken(), FakeClock()).read_one('one', 'Agents/AGENTS.md', {})


async def test_cancelling_one_inventory_reader_does_not_cancel_another_scope_peer() -> (
    None
):
    memory, clock = Memory(), FakeClock()
    memory.gate = asyncio.Event()
    assets = service(memory, clock)
    first = asyncio.create_task(assets.inventory('one', 'command', {'user': 'alice'}))
    second = asyncio.create_task(assets.inventory('one', 'command', {'user': 'alice'}))
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    memory.gate.set()
    assert [meta.name for meta in await second] == ['check', 'other']
    assert not assets.pending


async def test_cancelling_last_reader_reclaims_the_remote_refresh() -> None:
    memory, clock = Memory(), FakeClock()
    memory.gate = asyncio.Event()
    assets = service(memory, clock)
    task = asyncio.create_task(assets.inventory('one', 'command', {'user': 'alice'}))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert not assets.pending
    assert not assets.cache


async def test_dated_metadata_and_duplicate_skills_do_not_hide_the_inventory(
    tmp_path: Path,
) -> None:
    from app.features.skills.sources.assets import AssetSkillSource

    root = tmp_path / 'assets'
    for folder, front in (
        ('dated', 'name: dated\nupdated: 2026-10-18\nseen: [2026-10-18 09:30:00]'),
        ('first', 'name: twin'),
        ('second', 'name: twin'),
    ):
        (root / 'Agents/skills' / folder).mkdir(parents=True)
        (root / 'Agents/skills' / folder / 'SKILL.md').write_text(
            f'---\n{front}\ndescription: {folder}\n---\nbody',
        )
    assets = AssetsService(
        AssetsConfig(
            sources={
                'files': FilesystemAssetSourceConfig(kind='filesystem', root=root),
            },
        ),
        Memory(),
        FakeClock(),
        logger=make_logger(),
    )
    metas = {meta.name: meta for meta in await assets.inventory('files', 'skill', {})}
    assert sorted(metas) == ['dated', 'twin']
    assert metas['dated'].frontmatter['updated'] == '2026-10-18'
    assert metas['dated'].frontmatter['seen'] == ['2026-10-18T09:30:00']
    with pytest.raises(FileNotFoundError):
        await AssetSkillSource(assets, 'files').load('Agents/skills/gone/SKILL.md', {})


def test_non_string_frontmatter_keys_remain_invalid() -> None:
    from app.features.assets.sources import parse

    with pytest.raises(ValueError, match='frontmatter'):
        parse('files', 'command', 'x.md', '---\n1: one\n---\nbody')
