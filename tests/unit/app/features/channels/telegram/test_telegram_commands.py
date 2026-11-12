import json
from typing import override

import httpx2
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
from lib.http import HTTPClientContract
from tests.helpers.http import MockHTTP, Route

runner = CliRunner()

_SUPPORT_URL = 'https://api.telegram.org/botsupport-token/setMyCommands'
_RESEARCH_URL = 'https://api.telegram.org/botresearch-token/setMyCommands'
_SUPPORT_API = 'https://api.telegram.org/botsupport-token'
_RESEARCH_API = 'https://api.telegram.org/botresearch-token'


class FakeChannelsProvider(ProviderContract):
    def __init__(
        self,
        config: HuginnConfig,
        registry: CommandRegistry,
        http: MockHTTP,
    ) -> None:
        self.config = config
        self.registry = registry
        self.http = http

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.bind(HuginnConfig, instance=self.config)
        container.bind(CommandRegistry, instance=self.registry)
        container.bind(HTTPClientContract, instance=self.http.manager())


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


def invoke(
    http: MockHTTP,
    arguments: list[str],
    configured: HuginnConfig | None = None,
) -> Result:
    container = App().build(
        CommandProvider(),
        FakeChannelsProvider(configured or config(), registry(), http),
    )
    commands = container.resolve(CommandContract)
    commands.mount('telegram', cli)

    return runner.invoke(commands.app, ['telegram', *arguments])


def accepting(http: MockHTTP, url: str) -> Route:
    return http.post(url).mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': True}),
    )


def test_set_my_commands_publishes_the_registry_to_every_telegram_bot(
    http: MockHTTP,
) -> None:
    support = accepting(http, _SUPPORT_URL)
    research = accepting(http, _RESEARCH_URL)

    result = invoke(http, ['set-my-commands'])

    # NOTE: the menu is the same everywhere because it comes from one registry; each bot
    # needs its own call because the token is per bot
    assert result.exit_code == 0
    assert support.called
    assert research.called
    assert 'support_bot: published 2 commands' in result.output


def test_set_my_commands_can_publish_to_one_named_bot(http: MockHTTP) -> None:
    support = accepting(http, _SUPPORT_URL)
    research = accepting(http, _RESEARCH_URL)

    result = invoke(http, ['set-my-commands', 'support_bot'])

    assert result.exit_code == 0
    assert support.called
    assert not research.called


def test_the_published_menu_is_the_registry_in_listing_order(http: MockHTTP) -> None:
    support = accepting(http, _SUPPORT_URL)
    accepting(http, _RESEARCH_URL)

    invoke(http, ['set-my-commands', 'support_bot'])

    assert support.calls.last.request.content == (
        b'{"commands":[{"command":"help","description":"List the commands."},'
        b'{"command":"new","description":"Start over."}]}'
    )


def test_an_unknown_channel_name_is_reported_without_calling_telegram(
    http: MockHTTP,
) -> None:
    result = invoke(http, ['set-my-commands', 'nope_bot'])

    assert result.exit_code == 1
    assert 'nope_bot' in result.output


def test_set_webhook_registers_the_configured_base_url_with_the_channel_secret(
    http: MockHTTP,
) -> None:
    support = accepting(http, f'{_SUPPORT_API}/setWebhook')
    research = accepting(http, f'{_RESEARCH_API}/setWebhook')

    result = invoke(http, ['set-webhook'])

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


def test_set_webhook_takes_the_url_and_updates_from_the_command_line(
    http: MockHTTP,
) -> None:
    support = accepting(http, f'{_SUPPORT_API}/setWebhook')

    result = invoke(
        http,
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


def test_set_webhook_needs_a_url_from_somewhere(http: MockHTTP) -> None:
    support = accepting(http, f'{_SUPPORT_API}/setWebhook')

    result = invoke(
        http,
        ['set-webhook', 'support_bot'],
        config(telegram('support-token', base_url=None)),
    )

    assert result.exit_code == 1
    assert '--url' in result.output
    assert not support.called


def test_delete_webhook_keeps_pending_updates_unless_asked(http: MockHTTP) -> None:
    support = accepting(http, f'{_SUPPORT_API}/deleteWebhook')

    result = invoke(http, ['delete-webhook', 'support_bot'])

    assert result.exit_code == 0
    assert json.loads(support.calls.last.request.content) == {
        'drop_pending_updates': False,
    }
    assert 'support_bot: unregistered' in result.output


def test_delete_webhook_can_drop_the_queued_updates(http: MockHTTP) -> None:
    support = accepting(http, f'{_SUPPORT_API}/deleteWebhook')

    invoke(http, ['delete-webhook', 'support_bot', '--drop-pending'])

    assert json.loads(support.calls.last.request.content) == {
        'drop_pending_updates': True,
    }


def informing(http: MockHTTP, url: str, result: dict[str, object]) -> Route:
    return http.post(url).mock(
        return_value=httpx2.Response(200, json={'ok': True, 'result': result}),
    )


def test_webhook_info_prints_the_registration_and_its_last_failure(
    http: MockHTTP,
) -> None:
    informing(
        http,
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

    result = invoke(http, ['webhook-info', 'support_bot'])

    assert result.exit_code == 0
    assert 'url: https://huginn.example.com/webhooks/telegram' in result.output
    assert 'pending updates: 3' in result.output
    assert 'allowed updates: message' in result.output
    assert 'max connections: 40' in result.output
    # NOTE: the vendor reports the failure as a unix timestamp; the repository is what
    # turns it into something an operator can compare against a deploy time
    assert '2024-07-21T22:13:20+00:00' in result.output
    assert 'Wrong response from the webhook: 502' in result.output


def test_webhook_info_says_so_when_nothing_is_registered(http: MockHTTP) -> None:
    informing(
        http,
        f'{_SUPPORT_API}/getWebhookInfo',
        {'url': '', 'has_custom_certificate': False, 'pending_update_count': 0},
    )

    result = invoke(http, ['webhook-info', 'support_bot'])

    assert result.exit_code == 0
    assert 'url: (unregistered)' in result.output
    assert 'allowed updates: (vendor default)' in result.output
    assert 'last error' not in result.output
