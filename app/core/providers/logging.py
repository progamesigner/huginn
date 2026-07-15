from typing import override

from app.core.settings import AppSettings
from lib.container import ContainerContract, ProviderContract
from lib.observability import Logger, configure_logging, create_logger

from .settings import SettingsProvider


class LoggingProvider(ProviderContract):
    dependencies = (SettingsProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        configure_logging(
            develop=container.resolve(AppSettings).running_in_mode('dev'),
            capture=('uvicorn', 'uvicorn.error'),
            silence=('uvicorn.access',),
        )
        container.bind(Logger, factory=lambda _: create_logger('huginn'))
