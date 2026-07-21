from typing import override

from app.core.providers.database import DatabaseProvider
from app.core.providers.guid import GUIDProvider
from lib.container import ContainerContract, ProviderContract

from .repositories import IdentityRepository


class IdentitiesProvider(ProviderContract):
    dependencies = (DatabaseProvider, GUIDProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(IdentityRepository, constructor=IdentityRepository, scoped=True)
