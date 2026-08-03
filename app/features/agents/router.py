import logging
import re
from collections.abc import AsyncIterator, Mapping

from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    LLMContract,
    Message,
    TextChunk,
    TextSegment,
)

from .config import AgentConfig, resolve_model
from .exceptions import UnroutableError
from .types import RoutedAgent, RoutingPolicy, SelectionPolicy

logger = logging.getLogger(__name__)

_ROUTER_MAX_TOKENS = 32
_WORD = re.compile(r'[\w-]+')
_DEFAULT_PREAMBLE = (
    'You route a message to exactly one agent.\n'
    'Reply with only the agent name, exactly as written, and nothing else.'
)


async def _collect_text(stream: AsyncIterator[CompletionChunk]) -> str:
    return ''.join(
        [chunk.text async for chunk in stream if isinstance(chunk, TextChunk)],
    )


def _prompt(agents: Mapping[str, AgentConfig], preamble: str | None = None) -> str:
    lines = [
        preamble or _DEFAULT_PREAMBLE,
        '',
        'Agents:',
        *(f'- {name}: {agent.description}' for name, agent in agents.items()),
    ]
    return '\n'.join(lines)


def _parse_choice(reply: str, agents: Mapping[str, AgentConfig]) -> str | None:
    if (cleaned := reply.strip().strip('.\'"`')) in agents:
        return cleaned

    tokens = set(_WORD.findall(reply))
    matches = [name for name in agents if name in tokens]
    return matches[0] if len(matches) == 1 else None


class Router:
    agents: Mapping[str, AgentConfig]
    llm: LLMContract
    default_model: str | None
    prompt: str | None

    def __init__(
        self,
        agents: Mapping[str, AgentConfig],
        llm: LLMContract,
        default_model: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.agents = agents
        self.llm = llm
        self.default_model = default_model
        self.prompt = prompt

    async def resolve(
        self,
        policy: RoutingPolicy,
        chat_id: str,
        message: str,
    ) -> RoutedAgent:
        routable = self._routable(policy)
        if (name := policy.bindings.get(chat_id)) is not None:
            logger.info('Routed chat %s to agent %r via binding', chat_id, name)
            return self._routed(name, routable)

        if policy.agent is not None:
            logger.info(
                'Routed chat %s to agent %r via channel default',
                chat_id,
                policy.agent,
            )
            return self._routed(policy.agent, routable)

        if (selection := policy.selection) is None:
            raise UnroutableError(
                f'chat {chat_id!r} has no binding and the channel defines no router',
            )

        return self._routed(
            await self._select(selection, chat_id, message, routable),
            routable,
        )

    def _routable(self, policy: RoutingPolicy) -> Mapping[str, AgentConfig]:
        if not policy.allowed:
            return self.agents

        return {
            name: agent for name, agent in self.agents.items() if name in policy.allowed
        }

    def _routed(self, name: str, routable: Mapping[str, AgentConfig]) -> RoutedAgent:
        if name not in routable:
            raise UnroutableError(f'agent {name!r} is not routable for this caller')

        return RoutedAgent(name=name, config=routable[name])

    async def _select(
        self,
        selection: SelectionPolicy,
        chat_id: str,
        message: str,
        routable: Mapping[str, AgentConfig],
    ) -> str:
        choice = await self._ask(selection, message, routable)
        if choice is not None:
            logger.info('Router LLM chose agent %r for chat %s', choice, chat_id)
            return choice

        logger.info(
            'Router did not select a valid agent for chat %s; using fallback %r',
            chat_id,
            selection.fallback,
        )
        return selection.fallback

    async def _ask(
        self,
        selection: SelectionPolicy,
        message: str,
        routable: Mapping[str, AgentConfig],
    ) -> str | None:
        model = resolve_model(selection.model, self.default_model)
        if selection.kind is not None:
            provider = self.llm.provider_for(model)
            if provider is None or provider.adapter != selection.kind:
                logger.warning(
                    'Router model %r does not resolve to a %r provider; using fallback',
                    model,
                    selection.kind,
                )
                return None

        client = self.llm.client_for(model)
        if client is None:
            logger.warning('Router model %r has no provider; using fallback', model)
            return None

        request = CompletionRequest(
            model=model,
            messages=(
                Message(
                    role='system',
                    segments=(TextSegment(text=_prompt(routable, self.prompt)),),
                ),
                Message(role='user', segments=(TextSegment(text=message),)),
            ),
            max_completion_tokens=_ROUTER_MAX_TOKENS,
        )

        try:
            reply = await _collect_text(client.stream(request))
        except Exception:
            logger.exception('Router LLM call failed; using fallback')
            return None

        return _parse_choice(reply, routable)
