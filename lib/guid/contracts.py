from abc import ABC, abstractmethod
from uuid import UUID


class GUIDContract(ABC):
    @abstractmethod
    def generate(self) -> UUID: ...
