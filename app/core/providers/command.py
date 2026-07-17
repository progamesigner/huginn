from typing import override

from lib.command import CommandContract, CommandManager
from lib.container import ContainerContract, ProviderContract


class CommandProvider(ProviderContract):
    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(CommandContract, instance=CommandManager())
