import logging
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any, override

import pytest
import structlog

from lib.observability import Logger


class _EventText(logging.Filter):
    # NOTE: structlog hands the stdlib handler its event dict; caplog wants the text
    @override
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, dict):
            record.msg = str(record.msg.get('event', ''))
            record.args = ()
        return True


@contextmanager
def recording(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    # NOTE: building the app configures logging, which drops the handler caplog installed
    # before the test started -- so it goes back on once the container is wired
    root = logging.getLogger()
    text = _EventText()
    caplog.handler.addFilter(text)
    root.addHandler(caplog.handler)
    try:
        yield
    finally:
        root.removeHandler(caplog.handler)
        caplog.handler.removeFilter(text)


def _to_stdlib(
    _: object,
    __: str,
    event: MutableMapping[str, Any],
) -> tuple[tuple[object, ...], dict[str, Any]]:
    exc_info = event.pop('exc_info', None)
    return (event.pop('event'),), ({'exc_info': exc_info} if exc_info else {})


# NOTE: wraps a stdlib logger directly, so caplog sees the message whatever the global
# structlog setup happens to be when the test runs
def make_logger() -> Logger:
    return structlog.wrap_logger(
        logging.getLogger('tests'),
        processors=[structlog.stdlib.PositionalArgumentsFormatter(), _to_stdlib],
        wrapper_class=Logger,
    )
