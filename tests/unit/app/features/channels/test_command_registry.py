import pytest

from app.contracts.channels import InboundCommand, InboundMessage
from app.features.channels.exceptions import DuplicateCommandError
from app.features.channels.registry import CommandRegistry


def command(name: str, args: str = '') -> InboundCommand:
    return InboundCommand(name=name, args=args)


def inbound(text: str = '/new') -> InboundMessage:
    return InboundMessage(
        channel='support_bot',
        chat='4242',
        event='7',
        id='7',
        kind='telegram',
        text=text,
        user='77',
    )


def registry(*names: str) -> CommandRegistry:
    async def echo(given: InboundCommand, message: InboundMessage) -> str:
        return f'{given.name}:{given.args}:{message.chat}'

    built = CommandRegistry()
    for name in names:
        built.register(name, f'Do {name}.', echo)

    return built


async def test_dispatch_routes_to_the_handler_of_the_command() -> None:
    answer = await registry('new', 'help').dispatch(
        command('new', 'right now'),
        inbound(),
    )

    assert answer == 'new:right now:4242'


async def test_a_command_nobody_registered_answers_with_what_is_available() -> None:
    answer = await registry('help', 'new').dispatch(command('reboot'), inbound())

    # NOTE: the reply is the whole point of dispatching an unknown command -- silence
    # here is the same gap as an unanswered photo, an ack with nothing behind it
    assert answer == '/reboot is not a command I know. Available: /help, /new.'


async def test_an_empty_registry_still_answers_without_naming_nothing() -> None:
    answer = await registry().dispatch(command('help'), inbound())

    assert answer == '/help is not a command I know.'


def test_names_are_reported_in_a_stable_order() -> None:
    assert registry('new', 'help', 'compact').names() == ('compact', 'help', 'new')


def test_registering_a_command_twice_is_refused() -> None:
    async def other(given: InboundCommand, message: InboundMessage) -> str:
        return 'other'

    built = registry('new')

    with pytest.raises(DuplicateCommandError, match='new'):
        built.register('new', 'Do it again.', other)
