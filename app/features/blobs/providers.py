from typing import override

from app.core.providers.command import CommandProvider
from app.core.providers.config import ConfigProvider
from app.core.providers.database import DatabaseProvider
from app.core.providers.guid import GUIDProvider
from app.core.providers.temporal import ClockProvider
from lib.command import CommandContract
from lib.container import ContainerContract, ProviderContract

from .contracts import BlobReferenceRegistry, BlobReferenceSource, BlobServiceContract
from .services import BlobService


class BlobProvider(ProviderContract):
    dependencies = (
        ConfigProvider,
        ClockProvider,
        DatabaseProvider,
        GUIDProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(BlobServiceContract, constructor=BlobService, shared=True)
        container.bind(
            BlobReferenceRegistry,
            factory=lambda c: BlobReferenceRegistry(c.collect(BlobReferenceSource)),
            shared=True,
        )


class BlobCommandProvider(ProviderContract):
    dependencies = (BlobProvider, CommandProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .commands import cli

        container.resolve(CommandContract).mount('blobs', cli)
