from typing import override

from app.features.channels.providers import ChannelEndpointProvider
from app.features.messages.providers import MessageEndpointProvider
from app.features.probes.providers import ProbesEndpointProvider
from lib.container import ContainerContract, ProviderContract

from .telegram import TelegramEndpointProvider


class HuginnEndpointProvider(ProviderContract):
    dependencies = (
        ChannelEndpointProvider,
        MessageEndpointProvider,
        ProbesEndpointProvider,
        TelegramEndpointProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        pass
