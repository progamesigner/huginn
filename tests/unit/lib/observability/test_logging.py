import json
import logging
from collections.abc import Iterator

import pytest
import structlog

from lib.observability import Logger, configure_logging, create_logger, redact


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

    line = json.loads(capsys.readouterr().err)
    assert line['event'] == 'hello world'
    assert line['level'] == 'info'
    assert line['logger'] == 'vendor'
    assert 'timestamp' in line


def test_json_mode_renders_structlog_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(develop=False)

    create_logger('app').info('native', answer=42)

    line = json.loads(capsys.readouterr().err)
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
        json.loads(line) for line in capsys.readouterr().err.splitlines()
    )
    assert native['request_id'] == 'abc123'
    assert foreign['request_id'] == 'abc123'


def test_develop_renders_console_lines(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(develop=True)

    create_logger('app').info('pretty')

    out = capsys.readouterr().err
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
    assert json.loads(capsys.readouterr().err)['event'] == 'routed'


def test_silenced_loggers_emit_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(develop=False, silence=('vendor.noise',))

    silenced = logging.getLogger('vendor.noise')
    assert silenced.handlers == []
    assert not silenced.propagate

    silenced.info('dropped')
    assert capsys.readouterr().err == ''


def test_create_logger_returns_bound_logger() -> None:
    configure_logging(develop=False)

    assert isinstance(create_logger('app').bind(), Logger)


_BOT_TOKEN = '123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw4'  # noqa: S105


def test_httpx_request_lines_hide_the_telegram_bot_token(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(develop=False)

    logging.getLogger('httpx').info(
        'HTTP Request: %s %s "%s %d %s"',
        'POST',
        f'https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage',
        'HTTP/2',
        200,
        'OK',
    )

    line = json.loads(capsys.readouterr().err)
    assert _BOT_TOKEN not in line['event']
    assert 'api.telegram.org/bot[REDACTED]/sendMessage' in line['event']


@pytest.mark.parametrize('develop', [False, True])
def test_tracebacks_hide_the_telegram_bot_token(
    capsys: pytest.CaptureFixture[str],
    develop: bool,
) -> None:
    configure_logging(develop=develop)
    url = f'https://api.telegram.org/file/bot{_BOT_TOKEN}/photos/1.jpg'

    try:
        raise RuntimeError(f'download failed for {url}')
    except RuntimeError:
        create_logger('app').exception('Turn failed', token=_BOT_TOKEN)

    err = capsys.readouterr().err
    assert 'download failed' in err
    assert _BOT_TOKEN not in err
    assert _BOT_TOKEN.split(':')[1] not in err


@pytest.mark.parametrize(
    ('text', 'secret'),
    [
        (f'https://api.telegram.org/bot{_BOT_TOKEN}/getMe', _BOT_TOKEN),
        ('https://api.telegram.org/botshort-token/getMe', 'short-token'),
        (f'token is {_BOT_TOKEN}', _BOT_TOKEN),
        ("{'Authorization': 'Bearer sk-live-abc123'}", 'sk-live-abc123'),
        ("{'x-goog-api-key': 'AIzaSecret'}", 'AIzaSecret'),
        ('xi-api-key: el-secret', 'el-secret'),
        ('https://example.com/v1/models?key=AIzaSecret&alt=json', 'AIzaSecret'),
    ],
)
def test_redact_masks_known_secret_shapes(text: str, secret: str) -> None:
    assert secret not in redact(text)
    assert '[REDACTED]' in redact(text)


def test_redact_leaves_ordinary_text_alone() -> None:
    text = 'Acknowledged 42 after 0.120s at 12:30:45 via https://api.example.com/x'

    assert redact(text) == text
