from typing import override

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.providers.command import CommandProvider
from app.core.providers.config import ConfigProvider
from app.core.providers.database import DatabaseProvider
from app.core.providers.guid import GUIDProvider
from app.core.providers.temporal import ClockProvider
from lib.command import CommandContract
from lib.container import ContainerContract, ProviderContract
from lib.guid import GUIDContract
from lib.temporal import ClockContract

from .contracts import (
    BlobReferenceContract,
    BlobReferenceRegistry,
    BlobStoreContract,
    BlobStoreFactoryContract,
)
from .repositories import BlobRepository
from .services import BlobService


class BlobStoreFactory(BlobStoreFactoryContract):
    def __init__(self, guid: GUIDContract, clock: ClockContract) -> None:
        self.guid = guid
        self.clock = clock

    @override
    def for_session(self, session: AsyncSession) -> BlobStoreContract:
        return BlobRepository(session, self.guid, self.clock)


class BlobProvider(ProviderContract):
    dependencies = (
        ConfigProvider,
        ClockProvider,
        DatabaseProvider,
        GUIDProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(BlobStoreContract, constructor=BlobService, shared=True)
        container.bind(
            BlobStoreFactoryContract,
            constructor=BlobStoreFactory,
            shared=True,
        )
        container.bind(
            BlobReferenceRegistry,
            factory=lambda c: BlobReferenceRegistry(c.collect(BlobReferenceContract)),
            shared=True,
        )


class BlobCommandProvider(ProviderContract):
    dependencies = (BlobProvider, CommandProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .commands import cli

        container.resolve(CommandContract).mount('blobs', cli)
