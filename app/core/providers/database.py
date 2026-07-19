from typing import override

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract
from lib.database import DatabaseContract, DatabaseManager, ScopedSession

from .settings import SettingsProvider


class DatabaseProvider(ProviderContract):
    dependencies = (SettingsProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        settings = container.resolve(HuginnSettings)
        secret = settings.database_url
        manager = DatabaseManager(str(secret.get_secret_value()) if secret else '')

        container.bind(DatabaseContract, instance=manager)
        container.bind(ScopedSession, factory=lambda _: manager.scoped(), scoped=True)
        container.bind(
            AsyncSession,
            factory=lambda c: c.resolve(ScopedSession).session,
            scoped=True,
        )
