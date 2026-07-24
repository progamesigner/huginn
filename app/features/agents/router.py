import logging
import re
from collections.abc import AsyncIterator, Mapping

from app.features.channels.config import ChannelConfig, ChannelRouterConfig
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

logger = logging.getLogger(__name__)

_ROUTER_MAX_TOKENS = 32
_WORD = re.compile(r'[\w-]+')


async def _collect_text(stream: AsyncIterator[CompletionChunk]) -> str:
    return ''.join(
        [chunk.text async for chunk in stream if isinstance(chunk, TextChunk)]
    )


def _prompt(agents: Mapping[str, AgentConfig]) -> str:
    lines = [
        'You route a message to exactly one agent.',
        'Reply with only the agent name, exactly as written, and nothing else.',
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

    def __init__(
        self,
        agents: Mapping[str, AgentConfig],
        llm: LLMContract,
        default_model: str | None = None,
    ) -> None:
        self.agents = agents
        self.llm = llm
        self.default_model = default_model

    async def resolve(
        self,
        channel: ChannelConfig,
        chat_id: str,
        message: str,
    ) -> AgentConfig:
        if (name := channel.bindings.get(chat_id)) is not None:
            logger.info('Routed chat %s to agent %r via binding', chat_id, name)
            return self.agents[name]

        if channel.router is None:
            raise UnroutableError(
                f'chat {chat_id!r} has no binding and the channel defines no router'
            )

        return self.agents[await self._select(channel.router, chat_id, message)]

    async def _select(
        self,
        router: ChannelRouterConfig,
        chat_id: str,
        message: str,
    ) -> str:
        choice = await self._ask(router, message)
        if choice is not None:
            logger.info('Router LLM chose agent %r for chat %s', choice, chat_id)
            return choice

        logger.info(
            'Router did not select a valid agent for chat %s; using fallback %r',
            chat_id,
            router.fallback,
        )
        return router.fallback

    async def _ask(self, router: ChannelRouterConfig, message: str) -> str | None:
        model = resolve_model(router.model, self.default_model)
        client = self.llm.client_for(model)
        if client is None:
            logger.warning('Router model %r has no provider; using fallback', model)
            return None

        request = CompletionRequest(
            model=model,
            messages=(
                Message(
                    role='system',
                    segments=(TextSegment(text=_prompt(self.agents)),),
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

        return _parse_choice(reply, self.agents)
