from typer.testing import CliRunner

from lib.command import CommandManager, Typer, echo, resolve
from lib.container import Container

runner = CliRunner()


class Service:
    message: str

    def __init__(self, message: str) -> None:
        self.message = message


class ScopedService:
    disposed: bool

    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


def build_commands(message: str) -> Typer:
    commands = Typer()

    @commands.command()
    def hello() -> None:
        echo(message)

    return commands


def test_mount_exposes_named_command_group() -> None:
    manager = CommandManager(Container())
    manager.mount('demo', build_commands('hi from demo'))

    result = runner.invoke(manager.app, ['demo', 'hello'])

    assert result.exit_code == 0
    assert 'hi from demo' in result.output


def test_mounted_groups_stay_separate() -> None:
    manager = CommandManager(Container())
    manager.mount('first', build_commands('one'))
    manager.mount('second', build_commands('two'))

    first = runner.invoke(manager.app, ['first', 'hello'])
    second = runner.invoke(manager.app, ['second', 'hello'])

    assert 'one' in first.output
    assert 'two' in second.output


def test_resolve_uses_command_container() -> None:
    container = Container()
    container.bind(Service, instance=Service('from container'))
    manager = CommandManager(container)
    commands = Typer()

    @commands.command()
    def show() -> None:
        echo(resolve(Service).message)

    manager.mount('demo', commands)

    result = runner.invoke(manager.app, ['demo', 'show'])

    assert result.exit_code == 0
    assert result.output == 'from container\n'
