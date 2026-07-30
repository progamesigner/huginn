from typing import override

import httpx
import respx
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from app.core.application import App
from app.core.config.registry import HuginnConfig
from app.core.providers.command import CommandProvider
from app.features.channels.config import ChannelConfig, ChannelConfigList
from app.features.channels.registry import CommandRegistry
from app.features.channels.telegram.commands import cli
from app.features.channels.telegram.config import TelegramChannelConfig
from app.ports.channels import InboundCommand, InboundMessage
from lib.command import CommandContract
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract, HTTPClientManager

runner = CliRunner()

_SUPPORT_URL = 'https://api.telegram.org/botsupport-token/setMyCommands'
_RESEARCH_URL = 'https://api.telegram.org/botresearch-token/setMyCommands'


class FakeChannelsProvider(ProviderContract):
    def __init__(self, config: HuginnConfig, registry: CommandRegistry) -> None:
        self.config = config
        self.registry = registry

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(HuginnConfig, instance=self.config)
        container.bind(CommandRegistry, instance=self.registry)
        container.bind(HTTPClientContract, instance=HTTPClientManager())


def telegram(token: str) -> TelegramChannelConfig:
    return TelegramChannelConfig(
        kind='telegram',
        agent='scout',
        token=SecretStr(token),
        secret_token=SecretStr(f'{token}-secret'),
    )


def config() -> HuginnConfig:
    return HuginnConfig(
        sections={
            'channels': ChannelConfigList(
                {
                    'support_bot': telegram('support-token'),
                    'research_bot': telegram('research-token'),
                    # NOTE: another vendor's channel has no bot token at all, so the
                    # command has to walk past it rather than reach for one
                    'web_desk': ChannelConfig(kind='http', agent='scout'),
                },
            ),
        },
    )


def registry() -> CommandRegistry:
    async def handle(command: InboundCommand, message: InboundMessage) -> str:
        return 'done'

    built = CommandRegistry()
    built.register('new', 'Start over.', handle)
    built.register('help', 'List the commands.', handle)

    return built


def invoke(arguments: list[str]) -> Result:
    container = App().build(
        CommandProvider(),
        FakeChannelsProvider(config(), registry()),
    )
    commands = container.resolve(CommandContract)
    commands.mount('telegram', cli)

    return runner.invoke(commands.app, ['telegram', *arguments])


def accepting(url: str) -> respx.Route:
    return respx.post(url).mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': True}),
    )


@respx.mock
def test_set_my_commands_publishes_the_registry_to_every_telegram_bot() -> None:
    support = accepting(_SUPPORT_URL)
    research = accepting(_RESEARCH_URL)

    result = invoke(['set-my-commands'])

    # NOTE: the menu is the same everywhere because it comes from one registry; each bot
    # needs its own call because the token is per bot
    assert result.exit_code == 0
    assert support.called
    assert research.called
    assert 'support_bot: published 2 commands' in result.output


@respx.mock
def test_set_my_commands_can_publish_to_one_named_bot() -> None:
    support = accepting(_SUPPORT_URL)
    research = accepting(_RESEARCH_URL)

    result = invoke(['set-my-commands', 'support_bot'])

    assert result.exit_code == 0
    assert support.called
    assert not research.called


@respx.mock
def test_the_published_menu_is_the_registry_in_listing_order() -> None:
    support = accepting(_SUPPORT_URL)
    accepting(_RESEARCH_URL)

    invoke(['set-my-commands', 'support_bot'])

    assert support.calls.last.request.content == (
        b'{"commands":[{"command":"help","description":"List the commands."},'
        b'{"command":"new","description":"Start over."}]}'
    )


def test_an_unknown_channel_name_is_reported_without_calling_telegram() -> None:
    result = invoke(['set-my-commands', 'nope_bot'])

    assert result.exit_code == 1
    assert 'nope_bot' in result.output
