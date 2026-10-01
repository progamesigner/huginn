from typing import override

from app.core.providers.command import CommandProvider
from app.core.providers.redis import RedisProvider
from lib.command import CommandContract
from lib.container import ContainerContract, ProviderContract

from .services import RedisAuditor


class AuditorProvider(ProviderContract):
    dependencies = (RedisProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(RedisAuditor, constructor=RedisAuditor)


class AuditCommandProvider(ProviderContract):
    dependencies = (CommandProvider, AuditorProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .commands import cli

        container.resolve(CommandContract).mount('audit', cli)
