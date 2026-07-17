from typing import override

from lib.command import CommandManager
from lib.container import ContainerContract, ProviderContract


class CommandProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(CommandManager, instance=CommandManager())
