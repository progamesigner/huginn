from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class InboundMessage(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')

    channel: str
    chat: str
    event: str
    id: str
    text: str


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
    def reply(self, message: InboundMessage) -> ChannelReplyContract: ...
