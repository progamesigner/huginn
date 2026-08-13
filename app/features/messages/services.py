import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from datetime import timedelta

from app.contracts.auth import ApiClient
from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.types import (
    RoutingPolicy,
    SelectionPolicy,
    TurnPrompt,
    TurnRequest,
)
from lib.guid import GUIDContract
from lib.redis import RedisContract, RedisLock

from .exceptions import (
    ConversationBusyError,
    ForbiddenAgentError,
    UnroutableRequestError,
)
from .streams import QueuedReply, drain
from .types import OpenedEvent, TurnEvent

logger = logging.getLogger(__name__)

_LOCK_TTL = timedelta(seconds=30)


@dataclass(frozen=True)
class PreparedTurn:
    conversation: str
    request: TurnRequest
    lock: RedisLock


def _actor(client: ApiClient) -> str:
    return f'client:{client.key}'


def _policy(agent: str | None, client: ApiClient) -> RoutingPolicy:
    return RoutingPolicy(
        agent=agent,
        selection=(
            None
            if client.routing is None
            else SelectionPolicy(
                fallback=client.routing.fallback,
                model=client.routing.model,
                provider=client.routing.provider,
                allowed=client.routing.allowed,
            )
        ),
        allowed=client.agents,
    )


class ChatService:
    guid: GUIDContract
    redis: RedisContract
    turns: TurnHandlerContract
    ttl: timedelta

    def __init__(
        self,
        turns: TurnHandlerContract,
        redis: RedisContract,
        guid: GUIDContract,
        *,
        ttl: timedelta = _LOCK_TTL,
    ) -> None:
        self.guid = guid
        self.redis = redis
        self.turns = turns
        self.ttl = ttl

    async def prepare(
        self,
        client: ApiClient,
        message: str,
        *,
        conversation: str | None = None,
        agent: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> PreparedTurn:
        agent = agent or client.agent
        if agent is not None and agent not in client.agents:
            raise ForbiddenAgentError(f'this key may not route to agent {agent!r}')

        if agent is None and client.routing is None:
            raise UnroutableRequestError(
                'this key routes no message on its own; name an agent',
            )

        thread = conversation or str(self.guid.generate())
        lock = RedisLock(
            self.redis,
            f'messages:{client.key}:{thread}',
            ttl=self.ttl,
        )
        if not await lock.acquire():
            raise ConversationBusyError(f'conversation {thread!r} is already answering')

        return PreparedTurn(
            conversation=thread,
            request=TurnRequest(
                channel=client.key,
                chat=thread,
                policy=_policy(agent, client),
                prompts=(TurnPrompt(text=message, actor=_actor(client)),),
                denied_tools=client.denied_tools,
                max_completion_tokens=max_completion_tokens,
            ),
            lock=lock,
        )

    async def stream(self, prepared: PreparedTurn) -> AsyncGenerator[TurnEvent]:
        reply = QueuedReply()
        await reply.events.put(OpenedEvent(conversation=prepared.conversation))
        logger.info('Opened a turn on conversation %s', prepared.conversation)
        try:
            async with (
                prepared.lock.renew(),
                aclosing(drain(self.turns.handle(prepared.request, reply), reply)) as (
                    events
                ),
            ):
                async for event in events:
                    yield event
        finally:
            await prepared.lock.release()
