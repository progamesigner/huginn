import json
import logging
from collections.abc import Iterator

import pytest
import structlog

from lib.observability import Logger, configure_logging, create_logger


@pytest.fixture(autouse=True)
def isolate_logging() -> Iterator[None]:
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
    logging.getLogger().handlers = []


def test_json_mode_renders_stdlib_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(develop=False)

    logging.getLogger('vendor').info('hello %s', 'world')

    line = json.loads(capsys.readouterr().out)
    assert line['event'] == 'hello world'
    assert line['level'] == 'info'
    assert line['logger'] == 'vendor'
    assert 'timestamp' in line


def test_json_mode_renders_structlog_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(develop=False)

    create_logger('app').info('native', answer=42)

    line = json.loads(capsys.readouterr().out)
    assert line['event'] == 'native'
    assert line['answer'] == 42
    assert line['logger'] == 'app'


def test_contextvars_reach_native_and_foreign_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(develop=False)
    structlog.contextvars.bind_contextvars(request_id='abc123')

    create_logger('app').info('native')
    logging.getLogger('vendor').info('foreign')

    native, foreign = (
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    )
    assert native['request_id'] == 'abc123'
    assert foreign['request_id'] == 'abc123'


def test_develop_renders_console_lines(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(develop=True)

    create_logger('app').info('pretty')

    out = capsys.readouterr().out
    assert 'pretty' in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_captured_loggers_propagate_to_root(
    capsys: pytest.CaptureFixture[str],
) -> None:
    noisy = logging.getLogger('vendor.server')
    noisy.addHandler(logging.NullHandler())

    configure_logging(develop=False, capture=('vendor.server',))

    assert noisy.handlers == []
    assert noisy.propagate

    noisy.info('routed')
    assert json.loads(capsys.readouterr().out)['event'] == 'routed'


def test_silenced_loggers_emit_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(develop=False, silence=('vendor.noise',))

    silenced = logging.getLogger('vendor.noise')
    assert silenced.handlers == []
    assert not silenced.propagate

    silenced.info('dropped')
    assert capsys.readouterr().out == ''


def test_create_logger_returns_bound_logger() -> None:
    configure_logging(develop=False)

    assert isinstance(create_logger('app').bind(), Logger)
