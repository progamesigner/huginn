from typing import override

from app.core.providers.config import ConfigCommandProvider
from app.features.auditor.providers import AuditCommandProvider
from app.features.blobs.providers import BlobCommandProvider
from app.features.channels.telegram.providers import TelegramCommandProvider
from app.features.identities.providers import IdentityCommandProvider
from lib.container import ContainerContract, ProviderContract


class HuginnCommandProvider(ProviderContract):
    dependencies = (
        AuditCommandProvider,
        BlobCommandProvider,
        ConfigCommandProvider,
        IdentityCommandProvider,
        TelegramCommandProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
