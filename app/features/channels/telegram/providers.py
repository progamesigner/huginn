from datetime import timedelta
from typing import override

from app.contracts.channels import ChannelMessageHandlerContract
from app.core.config.registry import HuginnConfig
from app.core.providers.command import CommandProvider
from app.core.providers.config import ConfigProvider
from app.core.providers.http import HTTPClientProvider
from app.core.providers.redis import RedisProvider
from app.core.providers.router import RouterProvider
from app.core.providers.temporal import ClockProvider
from app.exts.telegram.gateways import build_gateway
from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.providers import TurnProvider
from app.features.channels.contracts import TranscriptionContract
from app.features.channels.registry import CommandRegistry
from app.features.conversations.providers import ConversationProvider
from app.features.identities.contracts import IdentityResolverContract
from app.features.identities.providers import IdentityProvider
from lib.command import CommandContract
from lib.concurrency import TaskSupervisor, TaskSupervisorContract
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract
from lib.redis import RedisContract
from lib.router import RouterContract
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
    http = container.resolve(HTTPClientContract)
    clock = container.resolve(ClockContract)
    return TelegramTurnProcessor(
        turns=container.resolve(TurnHandlerContract),
        commands=container.resolve(CommandRegistry),
        identities=container.resolve(IdentityResolverContract),
        gateway=lambda channel: build_gateway(http, channel.token, clock),
        transcription=container.resolve(TranscriptionContract),
    )


def _handler(container: ContainerContract) -> TelegramMessageHandler:
    return TelegramMessageHandler(
        config=container.resolve(HuginnConfig),
        redis=container.resolve(RedisContract),
        http=container.resolve(HTTPClientContract),
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


class TelegramEndpointProvider(ProviderContract):
    dependencies = (RouterProvider, TelegramProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        from app.exts.telegram.endpoints import router

        container.resolve(RouterContract).include(router)
