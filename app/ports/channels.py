from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict

type AttachmentKind = Literal['audio', 'image']


class InboundAttachment(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')

    kind: AttachmentKind
    mime: str
    reference: str
    unique: str
    duration: int | None = None
    height: int | None = None
    size: int | None = None
    width: int | None = None


class InboundMessage(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')

    channel: str
    chat: str
    event: str
    id: str
    text: str
    attachments: tuple[InboundAttachment, ...] = ()
    user: str | None = None


class ChannelMessageHandlerContract(ABC):
    @abstractmethod
    async def handle(self, message: InboundMessage) -> None: ...


class ChannelReplyContract(ABC):
    @abstractmethod
    async def abandon(self) -> None: ...

    @abstractmethod
    async def on_text(self, fragment: str) -> None: ...

    @abstractmethod
    async def finish(self) -> None: ...


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
