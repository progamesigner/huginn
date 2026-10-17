import hashlib
from collections.abc import AsyncGenerator, Callable
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
from app.features.identities.contracts import IdentityResolutionServiceContract
from app.features.identities.entities import LinkedPrincipal
from lib.guid import GUIDContract
from lib.observability import Logger
from lib.redis import RedisContract, RedisLock
from lib.result import Err, Ok, Result

from .attachments import AttachmentError, AttachmentIntake, Received, classify
from .entities import AttachmentEvent, OpenedEvent, TextEvent, TurnEvent
from .exceptions import (
    BlobNotFoundError,
    ConversationBusyError,
    ForbiddenAgentError,
    InvalidAttachmentError,
    UnroutableRequestError,
)
from .schemas import AttachmentInput
from .streams import QueuedReply, drain

type PrepareError = (
    AttachmentError
    | ConversationBusyError
    | ForbiddenAgentError
    | UnroutableRequestError
)
_LOCK_TTL = timedelta(seconds=30)


@dataclass(frozen=True)
class PreparedTurn:
    conversation: str
    request: TurnRequest
    lock: RedisLock
    owner: str | None = None


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
    intake: AttachmentIntake
    uploads: timedelta
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
        intake: AttachmentIntake,
        *,
        uploads: timedelta,
        identities: Callable[[], IdentityResolutionServiceContract] | None = None,
        ttl: timedelta = _LOCK_TTL,
        logger: Logger,
    ) -> None:
        self.logger = logger.bind(component='messages')
        self.blobs = blobs
        self.intake = intake
        self.uploads = uploads
        self.identities = identities
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
        attachments: tuple[AttachmentInput, ...] = (),
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
        channel = client.key
        principal = None
        if client.asserted_user is not None:
            if self.identities is None:
                raise RuntimeError('asserted identity resolution is unavailable')
            principal = await self.identities().resolve(
                'http',
                client.key,
                client.asserted_user,
            )
            digest = hashlib.sha256(client.asserted_user.encode()).hexdigest()
            channel = f'http:{len(client.key)}:{client.key}:user:{digest}'
        lock = RedisLock(
            self.redis,
            f'messages:{channel}:{thread}',
            ttl=self.ttl,
            logger=self.logger,
        )
        if not await lock.acquire():
            return Err(
                ConversationBusyError(f'conversation {thread!r} is already answering'),
            )

        # NOTE: reading an attachment can pay for a transcription, so a busy
        # conversation is turned away before any of it is read
        try:
            async with lock.renew():
                carried = await self.intake.take(client, attachments)
        except BaseException:
            await lock.release()
            raise
        if not isinstance(carried, tuple):
            await lock.release()
            return Err(carried)
        if attachments and not message and not carried:
            await lock.release()
            return Err(
                InvalidAttachmentError('nothing in this request could be read'),
            )

        return Ok(
            PreparedTurn(
                conversation=thread,
                request=TurnRequest(
                    channel=channel,
                    chat=thread,
                    policy=_policy(agent, client),
                    prompts=(
                        TurnPrompt(
                            text=message,
                            actor=principal.stable_key
                            if principal is not None
                            else _actor(client),
                            account=str(principal.user_id)
                            if isinstance(principal, LinkedPrincipal)
                            else None,
                            handle=principal.handle
                            if isinstance(principal, LinkedPrincipal)
                            else None,
                            memory_scope=principal.memory_scope
                            if isinstance(principal, LinkedPrincipal)
                            else None,
                            attachments=carried,
                        ),
                    ),
                    denied_tools=client.denied_tools,
                    max_completion_tokens=max_completion_tokens,
                ),
                lock=lock,
                owner=client.key,
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
                        if not await self._grant(
                            event,
                            prepared.owner or prepared.request.channel,
                        ):
                            yield TextEvent(
                                text='\n\nThe attachment could not be made available.',
                            )
                            continue
                    yield event
        finally:
            await prepared.lock.release()

    # NOTE: an upload is nobody's until a turn names it, so it lapses like a generated
    # file unless a recorded turn adopts it
    async def upload(
        self,
        client: ApiClient,
        data: bytes,
        name: str | None,
    ) -> Result[Blob, AttachmentError]:
        received = classify(data, name)
        if not isinstance(received, Received):
            return Err(received)

        blob = await self.blobs.save(
            received.mime,
            data,
            kind=received.kind,
            ttl=self.uploads,
        )
        await self.blobs.grant(blob.id, client.key, filename=name)
        self.logger.info('Stored an upload of %d bytes as %s', blob.size, blob.id)
        return Ok(blob.model_copy(update={'filename': name}))

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

        grant = await self.blobs.granted(id_, client.key)
        if grant is None:
            return missing

        found = await self.blobs.load(id_)
        if found is None or found.data is None:
            return missing

        return Ok(found.model_copy(update={'filename': grant.filename}))

    # NOTE: the file is the key's to fetch before the reader is told it exists
    async def _grant(self, event: AttachmentEvent, owner: str) -> bool:
        try:
            await self.blobs.grant(UUID(event.blob), owner, filename=event.name)
        except Exception:
            self.logger.exception('Could not grant %s to its reader', event.blob)
            return False

        return True
