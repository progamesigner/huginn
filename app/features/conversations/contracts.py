from abc import ABC, abstractmethod


class ConversationControlContract(ABC):
    @abstractmethod
    async def compact(self, channel: str, chat: str) -> bool: ...

    @abstractmethod
    async def reset(self, channel: str, chat: str) -> bool: ...
