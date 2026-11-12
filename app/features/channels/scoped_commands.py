import re
from dataclasses import dataclass, replace
from typing import override

from pydantic import JsonValue

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AGENTS
from app.features.agents.contracts import CommandInfo, ScopedCommandsServiceContract
from app.features.agents.entities import RoutingPolicy, TurnRequest, TurnScope
from app.features.assets.config import AssetsConfig
from app.features.assets.contracts import AssetsServiceContract
from app.features.assets.entities import AssetMeta
from lib.observability import Logger

_COMMAND = re.compile(r'^/([^\s/@]+)(?:@[A-Za-z0-9_]+)?(?:\s+([\s\S]*))?$')


@dataclass(frozen=True)
class ScopedCommand:
    meta: AssetMeta
    binding: dict[str, JsonValue]
    agents: frozenset[str]


def command_scope(request: TurnRequest, agent: str) -> TurnScope:
    prompt = request.prompts[-1]
    consistent = (
        len(
            {
                (item.account, item.handle, item.memory_scope)
                for item in request.prompts
            },
        )
        == 1
    )
    return TurnScope(
        agent=agent,
        actor=prompt.actor,
        channel=request.channel,
        chat=request.chat,
        account=prompt.account if consistent else None,
        handle=prompt.handle if consistent else None,
        memory_scope=prompt.memory_scope if consistent else None,
    )


class ScopedCommands(ScopedCommandsServiceContract):
    def __init__(
        self,
        config: HuginnConfig,
        settings: AssetsConfig,
        assets: AssetsServiceContract,
        reserved: frozenset[str],
        *,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='commands')
        self.config = config
        self.settings = settings
        self.assets = assets
        self.reserved = reserved

    def _agents(self, policy: RoutingPolicy) -> frozenset[str]:
        if policy.allowed:
            return policy.allowed
        if policy.selection is not None and policy.selection.allowed:
            return policy.selection.allowed
        if policy.agent is not None:
            return frozenset({policy.agent})
        return frozenset()

    async def _available(self, request: TurnRequest) -> dict[str, ScopedCommand]:
        if not request.prompts:
            return {}
        configured = self.config.section(AGENTS).root
        found = dict[str, ScopedCommand]()
        # A conflicting name is withheld everywhere; other commands stay usable.
        conflicts = set[str]()
        for name in sorted(self._agents(request.policy)):
            agent = configured.get(name)
            if agent is None or not agent.commands.enabled:
                continue
            source = agent.commands.source or 'default'
            binding = self.settings.binding(source, command_scope(request, name))
            if binding is None:
                continue
            for meta in await self.assets.inventory(source, 'command', binding):
                if agent.commands.allow and meta.name not in agent.commands.allow:
                    continue
                description = meta.frontmatter.get('description')
                if not isinstance(description, str) or not description.strip():
                    self.logger.warning(
                        'Skipping command with invalid description source=%s path=%s',
                        source,
                        meta.path,
                    )
                    continue
                if meta.name in self.reserved:
                    self.logger.warning(
                        'Skipping command %r in source %s: it conflicts with a built-in command',
                        meta.name,
                        source,
                    )
                    continue
                previous = found.get(meta.name)
                if previous is not None:
                    if (
                        previous.meta.source != source
                        or previous.meta.path != meta.path
                        or previous.binding != binding
                    ):
                        self.logger.warning(
                            'Withholding command %r: it is ambiguous across sources or scopes',
                            meta.name,
                        )
                        conflicts.add(meta.name)
                        continue
                    found[meta.name] = replace(
                        previous,
                        agents=previous.agents | {name},
                    )
                else:
                    found[meta.name] = ScopedCommand(meta, binding, frozenset({name}))
        # A metadata target must belong to the same resolved source and scope.
        return {
            name: command
            for name, command in found.items()
            if name not in conflicts
            and self._target(command.meta, command.agents) is not False
        }

    @staticmethod
    def _target(meta: AssetMeta, agents: frozenset[str]) -> str | bool | None:
        target = meta.frontmatter.get('agent')
        if target is None:
            return None
        return target if isinstance(target, str) and target in agents else False

    @override
    def enabled(self, policy: RoutingPolicy) -> bool:
        configured = self.config.section(AGENTS).root
        return any(
            (agent := configured.get(name)) is not None and agent.commands.enabled
            for name in self._agents(policy)
        )

    @override
    async def inventory(self, request: TurnRequest) -> tuple[CommandInfo, ...]:
        return tuple(
            CommandInfo(name, str(entry.meta.frontmatter['description']))
            for name, entry in sorted((await self._available(request)).items())
        )

    @override
    async def resolve(self, request: TurnRequest) -> TurnRequest:
        if not request.prompts or not self.enabled(request.policy):
            return request
        mixed = (
            len(
                {
                    (prompt.account, prompt.handle, prompt.memory_scope)
                    for prompt in request.prompts
                },
            )
            > 1
        )
        expanded = []
        allowed = self._agents(request.policy)
        target = None
        changed = False
        for prompt in request.prompts:
            match = _COMMAND.fullmatch(prompt.text) if prompt.origin == 'user' else None
            if match is None or match[1] in self.reserved:
                expanded.append(prompt)
                continue
            name, args = match[1], match[2] or ''
            single = request.model_copy(update={'prompts': (prompt,)})
            try:
                command = (await self._available(single)).get(name)
            except Exception as error:
                self.logger.warning('Could not list personal commands for /%s', name)
                raise ValueError(
                    'personal commands are temporarily unavailable; try again shortly',
                ) from error
            if command is None:
                # Leading-slash text is ordinary input unless it names a command;
                # channels with a command syntax report unknown names themselves.
                expanded.append(prompt)
                continue
            if mixed:
                raise ValueError(
                    'command batch mixes identity scopes; send commands separately',
                )
            asset = await self.assets.load(
                command.meta.source,
                'command',
                name,
                command.binding,
            )
            if asset is None:
                raise ValueError(f'command /{name} no longer exists')
            description = asset.meta.frontmatter.get('description')
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f'command /{name} metadata is invalid; refresh the inventory',
                )
            selected = self._target(asset.meta, command.agents)
            if selected is False:
                raise ValueError(f'command /{name} names an unauthorized agent')
            if isinstance(selected, str):
                if target is not None and target != selected:
                    raise ValueError('commands in this batch select different agents')
                target = selected
                allowed &= {selected}
            else:
                allowed &= command.agents
            body = (
                asset.body.replace('$ARGUMENTS', args)
                if '$ARGUMENTS' in asset.body
                else asset.body + ('\n\n' + args if args else '')
            )
            expanded.append(
                prompt.model_copy(
                    update={
                        'text': body,
                        'original_text': prompt.text,
                        'origin': 'template',
                    },
                ),
            )
            changed = True
        if not changed:
            return request
        if not allowed or (target is not None and target not in allowed):
            raise ValueError('commands in this batch have no common authorized agent')
        selection = request.policy.selection
        if selection is not None:
            candidates = allowed & (selection.allowed or allowed)
            if not candidates:
                raise ValueError('command is unavailable to this router')
            selection = selection.model_copy(
                update={
                    'allowed': candidates,
                    'fallback': selection.fallback
                    if selection.fallback in candidates
                    else sorted(candidates)[0],
                },
            )
        fixed = target or request.policy.agent
        if fixed is not None and fixed not in allowed:
            raise ValueError('command is unavailable to the selected agent')
        return request.model_copy(
            update={
                'prompts': tuple(expanded),
                'policy': RoutingPolicy(
                    agent=fixed,
                    allowed=allowed,
                    selection=selection,
                ),
            },
        )
