from typer.testing import CliRunner

from lib.command import CommandManager, Typer, echo

runner = CliRunner()


def build_commands(message: str) -> Typer:
    commands = Typer()

    @commands.command()
    def hello() -> None:
        echo(message)

    return commands


def test_mount_exposes_named_command_group() -> None:
    manager = CommandManager()
    manager.mount('demo', build_commands('hi from demo'))

    result = runner.invoke(manager.app, ['demo', 'hello'])

    assert result.exit_code == 0
    assert 'hi from demo' in result.output


def test_mounted_groups_stay_separate() -> None:
    manager = CommandManager()
    manager.mount('first', build_commands('one'))
    manager.mount('second', build_commands('two'))

    first = runner.invoke(manager.app, ['first', 'hello'])
    second = runner.invoke(manager.app, ['second', 'hello'])

    assert 'one' in first.output
    assert 'two' in second.output
