import json
from typing import override

import httpx
import respx
from pydantic import HttpUrl, SecretStr
from typer.testing import CliRunner, Result

from app.contracts.channels import InboundCommand, InboundMessage
from app.core.application import App
from app.core.config.registry import HuginnConfig
from app.core.providers.command import CommandProvider
from app.features.channels.config import ChannelConfig, ChannelConfigList
from app.features.channels.registry import Command, CommandRegistry
from app.features.channels.telegram.commands import cli
from app.features.channels.telegram.config import TelegramChannelConfig
from lib.command import CommandContract
from lib.container import ContainerContract, ProviderContract
from lib.http import HTTPClientContract, HTTPClientManager

runner = CliRunner()

_SUPPORT_URL = 'https://api.telegram.org/botsupport-token/setMyCommands'
_RESEARCH_URL = 'https://api.telegram.org/botresearch-token/setMyCommands'
_SUPPORT_API = 'https://api.telegram.org/botsupport-token'
_RESEARCH_API = 'https://api.telegram.org/botresearch-token'


class FakeChannelsProvider(ProviderContract):
    def __init__(self, config: HuginnConfig, registry: CommandRegistry) -> None:
        self.config = config
        self.registry = registry

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(HuginnConfig, instance=self.config)
        container.bind(CommandRegistry, instance=self.registry)
        container.bind(HTTPClientContract, instance=HTTPClientManager())


def telegram(
    token: str,
    base_url: str | None = 'https://huginn.example.com',
) -> TelegramChannelConfig:
    return TelegramChannelConfig(
        kind='telegram',
        agent='scout',
        token=SecretStr(token),
        secret_token=SecretStr(f'{token}-secret'),
        webhook_base_url=None if base_url is None else HttpUrl(base_url),
    )


def config(support: TelegramChannelConfig | None = None) -> HuginnConfig:
    return HuginnConfig(
        sections={
            'channels': ChannelConfigList(
                {
                    'support_bot': support or telegram('support-token'),
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

    built = CommandRegistry(
        [
            Command('new', 'Start over.', handle),
            Command('help', 'List the commands.', handle),
        ],
    )

    return built


def invoke(arguments: list[str], configured: HuginnConfig | None = None) -> Result:
    container = App().build(
        CommandProvider(),
        FakeChannelsProvider(configured or config(), registry()),
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


@respx.mock
def test_set_webhook_registers_the_configured_base_url_with_the_channel_secret() -> (
    None
):
    support = accepting(f'{_SUPPORT_API}/setWebhook')
    research = accepting(f'{_RESEARCH_API}/setWebhook')

    result = invoke(['set-webhook'])

    # NOTE: the path is the ext's to know; the command only ever hands over a base URL,
    # which is what keeps the registered URL and the receiving route from drifting apart
    assert result.exit_code == 0
    assert json.loads(support.calls.last.request.content) == {
        'url': 'https://huginn.example.com/webhooks/telegram',
        'secret_token': 'support-token-secret',
        'allowed_updates': ['message'],
        'drop_pending_updates': False,
    }
    assert research.called
    assert 'support_bot: registered https://huginn.example.com/webhooks/telegram' in (
        result.output
    )


@respx.mock
def test_set_webhook_takes_the_url_and_updates_from_the_command_line() -> None:
    support = accepting(f'{_SUPPORT_API}/setWebhook')

    result = invoke(
        [
            'set-webhook',
            'support_bot',
            '--url',
            'https://spare-otter.trycloudflare.com/',
            '--allowed-updates',
            'message',
            '--allowed-updates',
            'edited_message',
            '--drop-pending',
        ],
    )

    assert result.exit_code == 0
    assert json.loads(support.calls.last.request.content) == {
        'url': 'https://spare-otter.trycloudflare.com/webhooks/telegram',
        'secret_token': 'support-token-secret',
        'allowed_updates': ['message', 'edited_message'],
        'drop_pending_updates': True,
    }


@respx.mock
def test_set_webhook_needs_a_url_from_somewhere() -> None:
    support = accepting(f'{_SUPPORT_API}/setWebhook')

    result = invoke(
        ['set-webhook', 'support_bot'],
        config(telegram('support-token', base_url=None)),
    )

    assert result.exit_code == 1
    assert '--url' in result.output
    assert not support.called


@respx.mock
def test_delete_webhook_keeps_pending_updates_unless_asked() -> None:
    support = accepting(f'{_SUPPORT_API}/deleteWebhook')

    result = invoke(['delete-webhook', 'support_bot'])

    assert result.exit_code == 0
    assert json.loads(support.calls.last.request.content) == {
        'drop_pending_updates': False,
    }
    assert 'support_bot: unregistered' in result.output


@respx.mock
def test_delete_webhook_can_drop_the_queued_updates() -> None:
    support = accepting(f'{_SUPPORT_API}/deleteWebhook')

    invoke(['delete-webhook', 'support_bot', '--drop-pending'])

    assert json.loads(support.calls.last.request.content) == {
        'drop_pending_updates': True,
    }


def informing(url: str, result: dict[str, object]) -> respx.Route:
    return respx.post(url).mock(
        return_value=httpx.Response(200, json={'ok': True, 'result': result}),
    )


@respx.mock
def test_webhook_info_prints_the_registration_and_its_last_failure() -> None:
    informing(
        f'{_SUPPORT_API}/getWebhookInfo',
        {
            'url': 'https://huginn.example.com/webhooks/telegram',
            'has_custom_certificate': False,
            'pending_update_count': 3,
            'allowed_updates': ['message'],
            'last_error_date': 1_721_600_000,
            'last_error_message': 'Wrong response from the webhook: 502',
            'max_connections': 40,
        },
    )

    result = invoke(['webhook-info', 'support_bot'])

    assert result.exit_code == 0
    assert 'url: https://huginn.example.com/webhooks/telegram' in result.output
    assert 'pending updates: 3' in result.output
    assert 'allowed updates: message' in result.output
    assert 'max connections: 40' in result.output
    # NOTE: the vendor reports the failure as a unix timestamp; the repository is what
    # turns it into something an operator can compare against a deploy time
    assert '2024-07-21T22:13:20+00:00' in result.output
    assert 'Wrong response from the webhook: 502' in result.output


@respx.mock
def test_webhook_info_says_so_when_nothing_is_registered() -> None:
    informing(
        f'{_SUPPORT_API}/getWebhookInfo',
        {'url': '', 'has_custom_certificate': False, 'pending_update_count': 0},
    )

    result = invoke(['webhook-info', 'support_bot'])

    assert result.exit_code == 0
    assert 'url: (unregistered)' in result.output
    assert 'allowed updates: (vendor default)' in result.output
    assert 'last error' not in result.output
