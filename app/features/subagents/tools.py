import asyncio
import json
from collections.abc import Mapping
from typing import Annotated, override
from uuid import uuid4

from pydantic import Field, JsonValue, ValidationError
from structlog.contextvars import bound_contextvars

from app.features.agents.config import AgentConfig, SubagentDefinitionConfig
from app.features.agents.contracts import (
    ExecutionAccess,
    SubagentDefinition,
    SubagentServiceContract,
    SubagentStream,
    ToolHarvest,
    ToolOutput,
    ToolParams,
    ToolResult,
    ToolSource,
    TypedTool,
)
from app.features.agents.entities import TurnScope
from app.features.assets.config import AssetsConfig
from app.features.assets.contracts import AssetsServiceContract
from lib.observability import Logger


def vault_definition(
    front: Mapping[str, JsonValue],
    body: str,
) -> SubagentDefinitionConfig:
    if set(front) & {
        'provider',
        'budget',
        'mcp',
        'max_iterations',
        'max_completion_tokens',
        'turn_timeout',
    }:
        raise ValueError('vault definitions cannot set execution limits')
    selected = {
        key: value
        for key, value in front.items()
        if key in {'description', 'model', 'tools'}
    }
    return SubagentDefinitionConfig.model_validate_json(
        json.dumps({**selected, 'instructions': body}),
    )


class AgentParams(ToolParams):
    name: Annotated[str, Field(min_length=1)]
    task: Annotated[str, Field(min_length=1)]


class SubagentTool(TypedTool[AgentParams]):
    name = 'agent'
    params = AgentParams
    execution_policy = 'delegation'

    def __init__(
        self,
        agent: AgentConfig,
        assets: AssetsServiceContract,
        settings: AssetsConfig,
        runner: SubagentServiceContract,
        names: tuple[str, ...],
        descriptions: tuple[str, ...],
        *,
        logger: Logger,
    ) -> None:
        self.agent = agent
        self.assets = assets
        self.settings = settings
        self.runner = runner
        self.names = frozenset(names)
        self.description = (
            'Delegate a self-contained task. Only the final text and any produced files return; provide all needed context. Available:\n'
            + '\n'.join(descriptions)
        )
        self.logger = logger.bind(component='subagents')

    @override
    async def run(self, params: AgentParams) -> ToolOutput:
        stream = SubagentStream()
        text = await self._delegate(params, stream)
        # Files a child produced are delivered even when its answer is incomplete.
        return (
            ToolResult(text=text, artifacts=tuple(stream.artifacts))
            if stream.artifacts
            else text
        )

    async def _delegate(self, params: AgentParams, stream: SubagentStream) -> str:
        context = ExecutionAccess.current()
        if context is None or context.depth or not self.agent.subagents.enabled:
            return 'delegation denied: no parent execution or depth limit reached'
        if params.name not in self.names or (
            self.agent.subagents.allow and params.name not in self.agent.subagents.allow
        ):
            return f'unknown or disallowed subagent {params.name!r}'
        started = asyncio.get_running_loop().time()
        execution_id = uuid4().hex
        with bound_contextvars(subagent=params.name, subagent_execution=execution_id):
            status = 'failed'
            try:
                deadline = context.deadline - context.budget.min_iteration_budget
                async with asyncio.timeout_at(deadline), context.semaphore:
                    # Waiting and fetching the body both spend the parent's time.
                    definition = await self._load(params.name, context.scope)
                    if definition is None:
                        return 'subagent no longer available; refresh the inventory'
                    result = await self.runner.execute(
                        definition,
                        params.task,
                        context,
                        stream,
                    )
                    status = result.ending
                    return (
                        result.text
                        if result.completed
                        else f'subagent {result.ending}; no final answer returned'
                    )
            except TimeoutError:
                status = 'timeout'
                return 'subagent timed out before the parent answer deadline'
            except asyncio.CancelledError:
                status = 'cancelled'
                raise
            except ValueError, ValidationError:
                return 'subagent definition or budget is invalid; delegation refused'
            finally:
                self.logger.info(
                    'Subagent finished status=%s duration=%.3fs',
                    status,
                    asyncio.get_running_loop().time() - started,
                )

    async def _load(self, name: str, scope: TurnScope) -> SubagentDefinition | None:
        setting = self.agent.subagents
        config = setting.definitions.get(name)
        source = 'config'
        if config is None:
            source = setting.source or 'default'
            binding = self.settings.binding(source, scope)
            if binding is None:
                return None
            asset = await self.assets.load(source, 'subagent', name, binding)
            if asset is None:
                return None
            config = vault_definition(asset.meta.frontmatter, asset.body)
        self.logger.info('Selected subagent name=%s source=%s', name, source)
        return SubagentDefinition(
            name=name,
            source=source,
            instructions=config.instructions,
            model=config.model,
            tools=config.tools,
            max_iterations=config.max_iterations,
            max_completion_tokens=config.max_completion_tokens,
            turn_timeout=config.turn_timeout,
        )


class SubagentToolSource(ToolSource):
    def __init__(
        self,
        assets: AssetsServiceContract,
        settings: AssetsConfig,
        runner: SubagentServiceContract,
        *,
        logger: Logger,
    ) -> None:
        self.assets = assets
        self.settings = settings
        self.runner = runner
        self.logger = logger

    @override
    async def collect(self, agent: AgentConfig, scope: TurnScope) -> ToolHarvest:
        context = ExecutionAccess.current()
        if not agent.subagents.enabled or (context is not None and context.depth):
            return ToolHarvest()
        setting = agent.subagents
        descriptions = {
            name: entry.description for name, entry in setting.definitions.items()
        }
        source = setting.source or 'default'
        if source in self.settings.sources:
            binding = self.settings.binding(source, scope)
            if binding is not None:
                for meta in await self.assets.inventory(source, 'subagent', binding):
                    try:
                        description = vault_definition(
                            meta.frontmatter,
                            'inventory',
                        ).description
                    except ValueError:
                        self.logger.warning(
                            'Skipping invalid subagent metadata at %s',
                            meta.path,
                        )
                        continue
                    if meta.name in descriptions:
                        self.logger.warning(
                            'Config subagent overrides source=%s name=%s in current scope',
                            source,
                            meta.name,
                        )
                    else:
                        descriptions[meta.name] = description
        names = tuple(
            sorted(
                name
                for name in descriptions
                if not setting.allow or name in setting.allow
            ),
        )
        tool = SubagentTool(
            agent,
            self.assets,
            self.settings,
            self.runner,
            names,
            tuple(f'{name}: {descriptions[name]}' for name in names),
            logger=self.logger,
        )
        return ToolHarvest(tools=((tool.name, tool),))
