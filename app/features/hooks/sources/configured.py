import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import override

from pydantic import JsonValue

from app.features.agents.config import AgentConfig
from app.features.agents.contracts import HookHarvest, HookSource
from app.features.agents.entities import TurnScope, resolve_bindings
from app.features.hooks.config import (
    HookConfig,
    HTTPHookConfig,
    MCPHookConfig,
    ShellHookConfig,
)
from lib.http import HTTPClientContract
from lib.mcp import MCPClientContract
from lib.observability import Logger
from lib.temporal import ClockContract

from . import http, mcp, shell

type _Key = tuple[str, tuple[tuple[str, str], ...]]


class _HookUnavailableError(Exception):
    pass


@dataclass(frozen=True)
class _Cached:
    content: str
    expires_at: datetime | None


class ConfiguredHookSource(HookSource):
    def __init__(
        self,
        configs: Mapping[str, HookConfig],
        client: MCPClientContract,
        http_client: HTTPClientContract,
        clock: ClockContract,
        *,
        logger: Logger,
    ) -> None:
        self.configs = configs
        self.client = client
        self.http = http_client
        self.clock = clock
        self.logger = logger.bind(component='hooks')
        self.cache: dict[_Key, _Cached] = {}
        self.lock = asyncio.Lock()

    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> HookHarvest:
        contents = []
        unreachable = []
        for name in dict.fromkeys(agent.hooks):
            config = self.configs[name]
            bound = resolve_bindings({}, config.bind, scope)
            if bound is None:
                self.logger.info(
                    'Hook %s skipped; bound keys unavailable: %s',
                    name,
                    ', '.join(sorted(config.bind)),
                )
                continue
            try:
                content = await self._guarded_load(name, config, bound)
            except _HookUnavailableError:
                self.logger.warning('Hook %s unavailable (kind=%s)', name, config.kind)
                unreachable.append(name)
                continue
            if content:
                contents.append(content)
        return HookHarvest(
            content='\n\n'.join(contents) or None,
            unreachable=tuple(unreachable),
        )

    async def _guarded_load(
        self,
        name: str,
        config: HookConfig,
        bound: Mapping[str, JsonValue],
    ) -> str:
        try:
            return await self._load(name, config, bound)
        except Exception as error:
            # NOTE: the cause stays out of the log; a hook error can echo private state
            raise _HookUnavailableError(name) from error

    async def _load(
        self,
        name: str,
        config: HookConfig,
        bound: Mapping[str, JsonValue],
    ) -> str:
        key = (
            name,
            tuple(
                sorted(
                    (key, json.dumps(value, sort_keys=True))
                    for key, value in bound.items()
                ),
            ),
        )
        async with self.lock:
            now = self.clock.now()
            self.cache = {
                key: value
                for key, value in self.cache.items()
                if value.expires_at is None or value.expires_at > now
            }
            if cached := self.cache.get(key):
                return cached.content
            match config:
                case MCPHookConfig():
                    content = await mcp.read(self.client, config, bound)
                case HTTPHookConfig():
                    content = await http.read(self.http, config, bound)
                case ShellHookConfig():
                    content = await shell.read(config, bound)
            expires = (
                self.clock.now() + timedelta(seconds=config.cache_ttl)
                if config.cache_ttl is not None
                else None
            )
            self.cache[key] = _Cached(content, expires)
            return content
