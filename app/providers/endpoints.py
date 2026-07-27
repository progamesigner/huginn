from typing import override

from app.features.channels.providers import ChannelEndpointProvider
from app.features.probes.providers import ProbesEndpointProvider
from lib.container import ContainerContract, ProviderContract


class HuginnEndpointProvider(ProviderContract):
    dependencies = (ChannelEndpointProvider, ProbesEndpointProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
