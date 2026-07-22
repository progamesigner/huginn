from typing import override

from app.core.providers.database import DatabaseProvider
from app.core.providers.guid import GUIDProvider
from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract

from .exceptions import IdentityLinkSecretError
from .repositories import IdentityRepository
from .services import IdentityLinkingService, IdentityResolver
from .tokens import IdentityTokenCodec


def _token_codec(container: ContainerContract) -> IdentityTokenCodec:
    configured = container.resolve(HuginnSettings).identity_link_secret
    if configured is None:
        raise IdentityLinkSecretError('identity link secret is not configured')
    return IdentityTokenCodec(configured.get_secret_value().encode())


class IdentitiesProvider(ProviderContract):
    dependencies = (DatabaseProvider, GUIDProvider)

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
