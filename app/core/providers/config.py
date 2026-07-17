from pathlib import Path
from typing import override

from app.core.config import HuginnConfig, load_config
from app.core.config.commands import build_commands
from app.core.settings import HuginnSettings
from app.core.tools import ToolRegistry
from lib.command import CommandManager
from lib.container import ContainerContract, ProviderContract

from .command import CommandProvider
from .settings import SettingsProvider
from .tools import ToolRegistryProvider


class ConfigProvider(ProviderContract):
    dependencies = (SettingsProvider, ToolRegistryProvider)

    path: Path | None

    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(HuginnConfig, factory=self.build_config, shared=True)

    def build_config(self, container: ContainerContract) -> HuginnConfig:
        path = self.path or container.resolve(HuginnSettings).config_file
        return load_config(path, container.resolve(ToolRegistry))


class ConfigCommandProvider(ProviderContract):
    dependencies = (CommandProvider, SettingsProvider, ToolRegistryProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.resolve(CommandManager).mount('config', build_commands(container))
