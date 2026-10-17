from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr

type AttachmentKind = Literal['audio', 'document', 'image']
type ArtifactKind = Literal['audio', 'document', 'image', 'video']
ATTACHMENT_LIMITS: Mapping[AttachmentKind, int] = MappingProxyType(
    {
        'image': 5 * 1024 * 1024,
        'audio': 10 * 1024 * 1024,
        'document': 20 * 1024 * 1024,
    },
)
type ToolPhase = Literal['started', 'finished', 'failed']
type TurnEnding = Literal['complete', 'timeout', 'exhausted']


class InboundAttachment(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')

    kind: AttachmentKind
    mime: str
    reference: str
    unique: str
    name: str | None = None
    duration: int | None = None
    height: int | None = None
    size: int | None = None
    width: int | None = None


# NOTE: the bytes ride along because a transport cannot reach the blob store; the blob
# id is what a channel hands out when the bytes travel by another request
class ReplyArtifact(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')

    blob: str
    kind: ArtifactKind
    mime: str
    size: int
    data: bytes
    name: str | None = None


class InboundCommand(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')

    name: str
    args: str = ''


class InboundMessage(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')

    channel: str
    chat: str
    event: str
    id: str
    kind: str
    text: str
    attachments: tuple[InboundAttachment, ...] = ()
    command: InboundCommand | None = None
    user: str | None = None
    private: bool = False


class ChannelMessageHandlerContract(ABC):
    @abstractmethod
    async def handle(self, message: InboundMessage) -> None: ...


class ChannelReplyContract(ABC):
    @abstractmethod
    async def abandon(self) -> None: ...

    @abstractmethod
    async def finish(self) -> None: ...

    @abstractmethod
    async def on_artifact(self, artifact: ReplyArtifact) -> None: ...

    @abstractmethod
    async def on_routed(self, agent: str) -> None: ...

    @abstractmethod
    async def on_text(self, fragment: str) -> None: ...

    @abstractmethod
    async def on_tool(self, name: str, phase: ToolPhase) -> None: ...

    @abstractmethod
    async def unfinished(self, ending: TurnEnding) -> None: ...


class ChannelResolverContract(ABC):
    @abstractmethod
    def resolve(self, kind: str, credential: str) -> str | None: ...


class ChannelGatewayContract(ABC):
    @abstractmethod
    async def abandon(self, message: InboundMessage) -> None: ...

    @abstractmethod
    async def acknowledge(self, message: InboundMessage) -> None: ...

    @abstractmethod
    async def fetch(self, attachment: InboundAttachment) -> bytes: ...

    @abstractmethod
    def reply(self, message: InboundMessage) -> ChannelReplyContract: ...


# NOTE: a channel holds the credential, the transport knows what to build from it
class ChannelGatewayFactoryContract(ABC):
    @abstractmethod
    def gateway(self, token: SecretStr) -> ChannelGatewayContract: ...


class ChannelMenuContract(ABC):
    @abstractmethod
    async def set_commands(
        self,
        chat: str,
        member: str | None,
        commands: tuple[tuple[str, str], ...],
        language: str = '',
    ) -> None: ...

    @abstractmethod
    async def delete_commands(
        self,
        chat: str,
        member: str | None,
        language: str = '',
    ) -> None: ...
