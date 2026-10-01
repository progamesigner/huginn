from abc import ABC, abstractmethod

from lib.result import Result

from .entities import LinkedPrincipal, Principal
from .exceptions import InvalidIdentityLinkError


class IdentityLinkingServiceContract(ABC):
    @abstractmethod
    async def link(
        self,
        credential: str,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Result[LinkedPrincipal, InvalidIdentityLinkError]: ...


class IdentityResolutionServiceContract(ABC):
    @abstractmethod
    async def resolve(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Principal: ...
