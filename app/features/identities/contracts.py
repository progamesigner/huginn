from abc import ABC, abstractmethod

from .entities import LinkedPrincipal, Principal


class IdentityLinkingContract(ABC):
    @abstractmethod
    async def link(
        self,
        credential: str,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> LinkedPrincipal: ...


class IdentityResolverContract(ABC):
    @abstractmethod
    async def resolve(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Principal: ...
