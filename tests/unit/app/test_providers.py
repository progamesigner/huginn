from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from typer.testing import CliRunner

from app.core.application import App
from app.core.config.exceptions import UnknownSectionError
from app.core.config.registry import HuginnConfig, SectionRegistry
from app.core.providers.bootstrap import BootstrapProvider
from app.core.providers.router import RouterProvider
from app.core.providers.settings import SettingsProvider
from app.features.agents.contracts import (
    TranscriptStoreContract,
    TurnHandlerContract,
)
from app.features.agents.router import Router
from app.features.auditor.services import RedisAuditor
from app.features.channels.telegram.providers import TelegramProvider
from app.features.channels.telegram.service import TelegramMessageHandler
from app.features.identities.contracts import IdentityResolverContract
from app.features.tools.registry import ToolRegistry
from app.ports.channels import (
    ChannelMessageHandlerContract,
    ChannelResolverContract,
)
from app.providers import HuginnCommandProvider, HuginnEndpointProvider, HuginnProvider
from lib.command import CommandContract
from lib.mcp import MCPClientContract
from lib.router import RouterContract

runner = CliRunner()


def base() -> App:
    return App().build(SettingsProvider(), BootstrapProvider(), HuginnProvider())


def paths(container: App) -> set[str]:
    app = cast(FastAPI, container.resolve(RouterContract).app)
    return set(app.openapi()['paths'])


def test_base_provider_registers_every_feature_section() -> None:
    container = base()

    keys = {spec.key for spec in container.resolve(SectionRegistry)}

    assert {'agents', 'channels', 'mcp_servers', 'skills', 'tools'} <= keys
    assert 'websearch' in container.resolve(ToolRegistry).tools


def test_base_provider_binds_services_without_any_endpoint() -> None:
    container = base()

    # NOTE: the point of the *Provider / *EndpointProvider split
    assert not container.bound(RouterContract)
    assert container.bound(TelegramMessageHandler)
    assert container.bound(ChannelMessageHandlerContract)
    assert container.bound(ChannelResolverContract)
    assert container.bound(RedisAuditor)
    assert container.bound(MCPClientContract)


def test_base_provider_binds_the_turn_pipeline_end_to_end() -> None:
    container = base()

    # NOTE: the turn crosses three features and none of them may import the next one, so
    # what proves the pipeline exists is that every seam has an implementation bound
    assert container.bound(TurnHandlerContract)
    assert container.bound(TranscriptStoreContract)
    assert container.bound(IdentityResolverContract)
    assert container.bound(Router)


def test_endpoint_provider_mounts_every_route() -> None:
    container = App().build(
        SettingsProvider(),
        BootstrapProvider(),
        RouterProvider(),
        HuginnProvider(),
        HuginnEndpointProvider(),
    )

    assert {'/healthz', '/readyz', '/webhooks/telegram'} <= paths(container)


def test_endpoint_provider_pulls_the_base_wiring_its_routes_need() -> None:
    container = App().build(
        SettingsProvider(),
        BootstrapProvider(),
        RouterProvider(),
        HuginnEndpointProvider(),
    )

    assert container.bound(TelegramMessageHandler)
    assert container.bound(ChannelMessageHandlerContract)
    assert container.bound(ChannelResolverContract)


def test_command_provider_mounts_every_command_group() -> None:
    container = App().build(
        SettingsProvider(),
        BootstrapProvider(),
        HuginnProvider(),
        HuginnCommandProvider(),
    )

    result = runner.invoke(container.resolve(CommandContract).app, ['--help'])

    assert result.exit_code == 0
    for group in ('audit', 'config', 'identities'):
        assert group in result.output


def test_telegram_provider_binds_the_channel_ports_without_a_router() -> None:
    container = App().build(
        SettingsProvider(),
        BootstrapProvider(),
        TelegramProvider(),
    )

    assert container.bound(TelegramMessageHandler)
    assert container.bound(ChannelMessageHandlerContract)
    # NOTE: the resolver belongs to the channels feature, not to one vendor
    assert not container.bound(ChannelResolverContract)
    assert not container.bound(RouterContract)


def test_building_never_reads_the_config_file(tmp_path: Path) -> None:
    # NOTE: a broken config must still leave `huginn config validate` able to report it
    (tmp_path / 'huginn.yaml').write_text('nonexistent_section: {}\n')

    container = base()

    with pytest.raises(UnknownSectionError):
        container.resolve(HuginnConfig)
