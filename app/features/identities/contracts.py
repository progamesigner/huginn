from abc import ABC, abstractmethod

from lib.result import Result

from .entities import IssuedIdentityLinkInvite, LinkedPrincipal, Principal
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


class IdentityTargetServiceContract(ABC):
    @abstractmethod
    def resolve(self, target: str) -> tuple[str, str]: ...


class IdentityManagementServiceContract(ABC):
    @abstractmethod
    async def register(
        self,
        credential: str,
        provider: str,
        realm: str,
        external: str,
    ) -> Result[LinkedPrincipal | None, InvalidIdentityLinkError]: ...

    @abstractmethod
    async def claim(
        self,
        provider: str,
        realm: str,
        external: str,
        handle: str,
        *,
        credential: str | None = None,
        display_name: str | None = None,
    ) -> Result[LinkedPrincipal, InvalidIdentityLinkError]: ...

    @abstractmethod
    async def invite(
        self,
        provider: str,
        realm: str,
        external: str,
        target: str,
    ) -> IssuedIdentityLinkInvite: ...

    @abstractmethod
    async def display_name(
        self,
        provider: str,
        realm: str,
        external: str,
        name: str,
    ) -> None: ...
