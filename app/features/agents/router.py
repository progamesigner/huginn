import logging
from collections.abc import Mapping

from .config import AgentConfig, ChannelConfig
from .exceptions import UnroutableError

logger = logging.getLogger(__name__)


class Router:
    agents: Mapping[str, AgentConfig]

    def __init__(self, agents: Mapping[str, AgentConfig]) -> None:
        self.agents = agents

    async def resolve(
        self,
        channel: ChannelConfig,
        chat_id: str,
        message: str,
    ) -> AgentConfig:
        if (name := channel.bindings.get(chat_id)) is not None:
            logger.info('Routed chat %s to agent %r via binding', chat_id, name)
            return self.agents[name]

        if channel.router is not None:
            logger.info(
                'Chat %s has no binding; routing to fallback agent %r',
                chat_id,
                channel.router.fallback,
            )
            return self.agents[channel.router.fallback]

        raise UnroutableError(
            f'chat {chat_id!r} has no binding and the channel defines no router'
        )
