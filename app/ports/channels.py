from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class InboundMessage:
    channel: str
    chat: str
    event: str
    raw: bytes


class ChannelMessageHandlerContract(ABC):
    @abstractmethod
    async def handle(self, message: InboundMessage) -> None: ...


class ChannelReplyContract(ABC):
    @abstractmethod
    async def on_text(self, fragment: str) -> None: ...

    @abstractmethod
    async def finish(self) -> None: ...


class ChannelResolverContract(ABC):
    @abstractmethod
    def resolve(self, kind: str, credential: str) -> str | None: ...


class ChannelGatewayContract(ABC):
    @abstractmethod
    async def acknowledge(self, message: InboundMessage) -> None: ...

    @abstractmethod
    def reply(self, message: InboundMessage) -> ChannelReplyContract: ...
