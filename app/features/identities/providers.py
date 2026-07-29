from typing import override

from app.core.providers.command import CommandProvider
from app.core.providers.database import DatabaseProvider
from app.core.providers.guid import GUIDProvider
from app.core.providers.temporal import ClockProvider
from app.core.settings import HuginnSettings
from lib.command import CommandContract
from lib.container import ContainerContract, ProviderContract

from .contracts import IdentityResolverContract
from .exceptions import IdentityLinkSecretError
from .repositories import IdentityRepository
from .services import IdentityLinkingService, IdentityResolver, PrincipalResolver
from .tokens import IdentityTokenCodec


def _token_codec(container: ContainerContract) -> IdentityTokenCodec:
    configured = container.resolve(HuginnSettings).identity_link_secret
    if configured is None:
        raise IdentityLinkSecretError('identity link secret is not configured')
    return IdentityTokenCodec(configured.get_secret_value().encode())


class IdentityProvider(ProviderContract):
    dependencies = (DatabaseProvider, GUIDProvider, ClockProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(IdentityRepository, constructor=IdentityRepository, scoped=True)
        container.bind(IdentityTokenCodec, factory=_token_codec, scoped=True)
        container.bind(
            IdentityLinkingService,
            constructor=IdentityLinkingService,
            scoped=True,
        )

        container.bind(IdentityResolver, constructor=IdentityResolver, scoped=True)
        container.bind(
            IdentityResolverContract,
            constructor=PrincipalResolver,
            shared=True,
        )


class IdentityCommandProvider(ProviderContract):
    dependencies = (CommandProvider, IdentityProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .commands import cli

        container.resolve(CommandContract).mount('identities', cli)
