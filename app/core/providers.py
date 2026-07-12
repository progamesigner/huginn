from typing import cast, override

from app.core.application import App
from app.core.settings import AppSettings
from lib.container import ContainerContract, ProviderContract
from lib.observability import Logger, configure_logging, create_logger


class SettingsProvider(ProviderContract):
    @override
    def register(self, container: ContainerContract) -> None:
        container.bind(AppSettings, instance=AppSettings())


class LoggingProvider(ProviderContract):
    dependencies = (SettingsProvider,)

    @override
    def register(self, container: ContainerContract) -> None:
        configure_logging(
            develop=cast(App, container).running_in_mode('dev'),
            capture=('uvicorn', 'uvicorn.error'),
            silence=('uvicorn.access',),
        )
        container.bind(Logger, factory=lambda _: create_logger('huginn'))
