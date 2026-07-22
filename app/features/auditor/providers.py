from typing import override

from app.core.providers.command import CommandProvider
from app.core.providers.redis import RedisProvider
from lib.command import CommandContract
from lib.container import ContainerContract, ProviderContract

from .services import RedisAuditor


class AuditCommandProvider(ProviderContract):
    dependencies = (CommandProvider, RedisProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .commands import cli

        container.bind(RedisAuditor, constructor=RedisAuditor, scoped=True)
        container.resolve(CommandContract).mount('audit', cli)
