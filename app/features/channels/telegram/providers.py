from datetime import timedelta
from typing import override

from app.contracts.channels import (
    ChannelGatewayFactoryContract,
    ChannelMessageHandlerContract,
)
from app.core.config.registry import HuginnConfig
from app.core.providers.command import CommandProvider
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.core.providers.redis import RedisProvider
from app.core.providers.temporal import ClockProvider
from app.features.agents.contracts import TurnServiceContract
from app.features.agents.providers import TurnProvider
from app.features.channels.contracts import TranscriptionServiceContract
from app.features.channels.registry import CommandRegistry
from app.features.conversations.providers import ConversationProvider
from app.features.identities.contracts import IdentityResolutionServiceContract
from app.features.identities.providers import IdentityProvider
from lib.command import CommandContract
from lib.concurrency import TaskSupervisor, TaskSupervisorContract
from lib.container import ContainerContract, ProviderContract
from lib.redis import RedisContract
from lib.temporal import ClockContract

from .actor import TelegramChatActor
from .service import TelegramMessageHandler
from .turns import TelegramTurnProcessor

_TURN_LIMIT = 32
_TURN_GRACE = timedelta(seconds=60)


def _actor(container: ContainerContract) -> TelegramChatActor:
    return TelegramChatActor(
        container.resolve(RedisContract),
        container.resolve(TelegramTurnProcessor).process,
    )


def _turns(container: ContainerContract) -> TelegramTurnProcessor:
    gateways = container.resolve(ChannelGatewayFactoryContract)
    return TelegramTurnProcessor(
        turns=container.resolve(TurnServiceContract),
        commands=container.resolve(CommandRegistry),
        identities=container.resolve(IdentityResolutionServiceContract),
        gateway=lambda channel: gateways.gateway(channel.token),
        transcription=container.resolve(TranscriptionServiceContract),
    )


def _handler(container: ContainerContract) -> TelegramMessageHandler:
    return TelegramMessageHandler(
        config=container.resolve(HuginnConfig),
        redis=container.resolve(RedisContract),
        gateways=container.resolve(ChannelGatewayFactoryContract),
        clock=container.resolve(ClockContract),
        actor=container.resolve(TelegramChatActor),
        supervisor=container.resolve(TaskSupervisorContract),
    )


class TelegramProvider(ProviderContract):
    dependencies = (
        ClockProvider,
        ConfigProvider,
        ConversationProvider,
        HTTPClientProvider,
        IdentityProvider,
        RedisProvider,
        TurnProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(
            TaskSupervisorContract,
            instance=TaskSupervisor(limit=_TURN_LIMIT, grace=_TURN_GRACE),
        )
        container.bind(TelegramTurnProcessor, factory=_turns)
        container.bind(TelegramChatActor, factory=_actor)
        container.bind(TelegramMessageHandler, factory=_handler)
        container.bind(ChannelMessageHandlerContract, factory=_handler)


class TelegramCommandProvider(ProviderContract):
    dependencies = (CommandProvider, ConfigProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from .commands import cli

        container.resolve(CommandContract).mount('telegram', cli)
