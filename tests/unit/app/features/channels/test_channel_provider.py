from app.features.channels.registry import CommandRegistry
from app.providers.contributions import ChannelContributions
from tests.helpers.settings import app_with, make_settings

_URL = 'postgresql+asyncpg://user:pass@localhost/huginn'


async def test_the_channel_provider_registers_the_command_set() -> None:
    settings = make_settings(database_url=_URL)
    container = app_with(settings).build(ChannelContributions())

    registry = container.resolve(CommandRegistry)

    assert registry.names() == (
        'compact',
        'displayname',
        'help',
        'iam',
        'identity',
        'invite',
        'new',
        'register',
        'start',
    )
    assert all(command.description for command in registry.listing())

    await container.dispose()


async def test_registering_the_commands_resolves_nothing_yet() -> None:
    # NOTE: providers run before the config file is read, so a handler that resolved its
    # dependencies at registration would make building the app need a database
    settings = make_settings()
    container = app_with(settings).build(ChannelContributions())

    assert container.resolve(CommandRegistry).names()

    await container.dispose()
