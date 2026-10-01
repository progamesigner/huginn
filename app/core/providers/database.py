from typing import override

from app.core.settings import HuginnSettings
from lib.container import ContainerContract, ProviderContract
from lib.database import DatabaseContract, DatabaseError, DatabaseManager

from .settings import SettingsProvider


class DatabaseProvider(ProviderContract):
    dependencies = (SettingsProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        def _factory(container: ContainerContract) -> DatabaseManager:
            configured = container.resolve(HuginnSettings).database_urls
            if not configured:
                raise DatabaseError('database urls are not configured')

            return DatabaseManager(
                {name: str(url.get_secret_value()) for name, url in configured.items()},
            )

        container.bind(DatabaseContract, factory=_factory, shared=True)
