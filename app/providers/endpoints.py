from typing import override

from app.features.channels.providers import ChannelEndpointProvider
from app.features.messages.providers import MessageEndpointProvider
from app.features.probes.providers import ProbesEndpointProvider
from lib.container import ContainerContract, ProviderContract


class HuginnEndpointProvider(ProviderContract):
    dependencies = (
        ChannelEndpointProvider,
        MessageEndpointProvider,
        ProbesEndpointProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
