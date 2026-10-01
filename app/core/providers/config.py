from typing import override

from app.core.config.loader import load_config
from app.core.config.providers import PROVIDERS
from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.settings import HuginnSettings
from lib.command import CommandContract
from lib.container import ContainerContract, ProviderContract

from .command import CommandProvider
from .settings import SettingsProvider


class ConfigProvider(ProviderContract):
    dependencies = (SettingsProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(SectionRegistry, instance=SectionRegistry())
        container.bind(HuginnConfig, factory=self.build_config, shared=True)

    def build_config(self, container: ContainerContract) -> HuginnConfig:
        return load_config(
            container.resolve(HuginnSettings).config_file,
            container.resolve(SectionRegistry),
        )


class ProvidersConfigProvider(ProviderContract):
    dependencies = (ConfigProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.resolve(SectionRegistry).register(PROVIDERS)


class ConfigCommandProvider(ProviderContract):
    dependencies = (CommandProvider, ConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from app.core.config.commands import cli

        container.resolve(CommandContract).mount('config', cli)
