from abc import ABC, abstractmethod

from .types import Principal


class IdentityResolverContract(ABC):
    @abstractmethod
    async def resolve(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Principal: ...
