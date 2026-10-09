from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

from app.contracts.auth import ApiClient
from app.features.agents.contracts import TurnServiceContract
from app.features.agents.entities import (
    RoutingPolicy,
    SelectionPolicy,
    ShadowPolicy,
    TurnPrompt,
    TurnRequest,
)
from app.features.blobs.contracts import BlobServiceContract
from app.features.blobs.entities import Blob
from lib.guid import GUIDContract
from lib.observability import Logger
from lib.redis import RedisContract, RedisLock
from lib.result import Err, Ok, Result

from .entities import AttachmentEvent, OpenedEvent, TurnEvent
from .exceptions import (
    BlobNotFoundError,
    ConversationBusyError,
    ForbiddenAgentError,
    UnroutableRequestError,
)
from .streams import QueuedReply, drain

type PrepareError = ConversationBusyError | ForbiddenAgentError | UnroutableRequestError
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
                engine=client.routing.engine,
                history_turns=client.routing.history_turns,
                model=client.routing.model,
                provider=client.routing.provider,
                allowed=client.routing.allowed,
                min_confidence=client.routing.min_confidence,
                min_margin=client.routing.min_margin,
                shadow=(
                    None
                    if client.routing.shadow_model is None
                    else ShadowPolicy(
                        model=client.routing.shadow_model,
                        provider=client.routing.shadow_provider,
                    )
                ),
            )
        ),
        allowed=client.agents,
    )


class ChatService:
    logger: Logger
    blobs: BlobServiceContract
    guid: GUIDContract
    redis: RedisContract
    turns: TurnServiceContract
    ttl: timedelta

    def __init__(
        self,
        turns: TurnServiceContract,
        redis: RedisContract,
        guid: GUIDContract,
        blobs: BlobServiceContract,
        *,
        ttl: timedelta = _LOCK_TTL,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='messages')
        self.blobs = blobs
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
    ) -> Result[PreparedTurn, PrepareError]:
        agent = agent or client.agent
        if agent is not None and agent not in client.agents:
            return Err(
                ForbiddenAgentError(f'this key may not route to agent {agent!r}'),
            )

        if agent is None and client.routing is None:
            return Err(
                UnroutableRequestError(
                    'this key routes no message on its own; name an agent',
                ),
            )

        thread = conversation or str(self.guid.generate())
        lock = RedisLock(
            self.redis,
            f'messages:{client.key}:{thread}',
            ttl=self.ttl,
            logger=self.logger,
        )
        if not await lock.acquire():
            return Err(
                ConversationBusyError(f'conversation {thread!r} is already answering'),
            )

        return Ok(
            PreparedTurn(
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
            ),
        )

    async def stream(self, prepared: PreparedTurn) -> AsyncGenerator[TurnEvent]:
        reply = QueuedReply()
        await reply.events.put(OpenedEvent(conversation=prepared.conversation))
        self.logger.info('Opened a turn on conversation %s', prepared.conversation)
        try:
            async with (
                prepared.lock.renew(),
                aclosing(
                    drain(
                        self.turns.handle(prepared.request, reply),
                        reply,
                        logger=self.logger,
                    ),
                ) as (events),
            ):
                async for event in events:
                    if isinstance(event, AttachmentEvent):
                        await self._grant(event, prepared.request.channel)
                    yield event
        finally:
            await prepared.lock.release()

    # NOTE: absent and forbidden answer alike, so an id says nothing about whether
    # the blob behind it exists
    async def download(
        self,
        client: ApiClient,
        blob: str,
    ) -> Result[Blob, BlobNotFoundError]:
        missing = Err(BlobNotFoundError(f'no blob {blob!r} for this key'))
        try:
            id_ = UUID(blob)
        except ValueError:
            return missing

        if not await self.blobs.granted(id_, client.key):
            return missing

        found = await self.blobs.load(id_)
        if found is None or found.data is None:
            return missing

        return Ok(found)

    # NOTE: the file is the key's to fetch before the reader is told it exists
    async def _grant(self, event: AttachmentEvent, owner: str) -> None:
        try:
            await self.blobs.grant(UUID(event.blob), owner)
        except Exception:
            self.logger.exception('Could not grant %s to its reader', event.blob)
