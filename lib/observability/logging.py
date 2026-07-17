import logging
import sys
from collections.abc import Sequence

import structlog
from structlog.stdlib import BoundLogger as Logger


def configure_logging(
    *,
    develop: bool,
    capture: Sequence[str] = (),
    silence: Sequence[str] = (),
) -> None:
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt='iso', utc=True),
    ]

    if develop:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=Logger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.DEBUG if develop else logging.INFO)

    for name in capture:
        captured = logging.getLogger(name)
        captured.handlers = []
        captured.propagate = True

    for name in silence:
        silenced = logging.getLogger(name)
        silenced.handlers = []
        silenced.propagate = False


def create_logger(name: str) -> Logger:
    return structlog.stdlib.get_logger(name)
