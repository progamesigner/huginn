import logging
import re
from typing import override

import structlog

REDACTED = '[REDACTED]'

# NOTE: applied to the rendered line, so tracebacks and every logger are covered
_PATTERNS = (
    (re.compile(r'(api\.telegram\.org/(?:file/)?bot)[^/\s\'"]+'), rf'\1{REDACTED}'),
    # bare telegram bot token: <bot id>:<secret>
    (re.compile(r'(?<![0-9])[0-9]{6,}:[A-Za-z0-9_-]{30,}'), REDACTED),
    (re.compile(r'(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]+'), rf'\1{REDACTED}'),
    (
        re.compile(
            r'(?i)(\b(?:x-goog-api-key|xi-api-key|x-api-key|authorization)'
            r'[\'"]?\s*[:=]\s*[\'"]?)(?!bearer\s)[^\s\'",}]+',
        ),
        rf'\1{REDACTED}',
    ),
    (
        re.compile(r'(?i)([?&](?:key|api_key|access_token|token)=)[^&\s\'"]+'),
        rf'\1{REDACTED}',
    ),
)


def redact(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    return text


class RedactingFormatter(structlog.stdlib.ProcessorFormatter):
    @override
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))
