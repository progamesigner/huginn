import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest


@contextmanager
def recording(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    # NOTE: building the app configures logging, which drops the handler caplog installed
    # before the test started -- so it goes back on once the container is wired
    root = logging.getLogger()
    root.addHandler(caplog.handler)
    try:
        yield
    finally:
        root.removeHandler(caplog.handler)
