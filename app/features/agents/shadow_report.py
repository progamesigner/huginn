import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .eval_report import percentile

_SHADOW = re.compile(r'Router shadow engine=decision (?P<fields>.*)$')
_FIELD = re.compile(r'(\w+)=(\S+)')
_BUCKETS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))


@dataclass(frozen=True)
class ShadowRecord:
    primary: str
    choice: str | None
    confidence: float | None
    elapsed_ms: float
    reason: str | None


def _optional(value: str) -> str | None:
    return None if value == 'None' else value


def _record(message: str) -> ShadowRecord | None:
    if (match := _SHADOW.search(message)) is None:
        return None

    fields = dict(_FIELD.findall(match['fields']))
    confidence = _optional(fields.get('confidence', 'None'))
    return ShadowRecord(
        primary=fields['primary'],
        choice=_optional(fields.get('choice', 'None')),
        confidence=None if confidence is None else float(confidence),
        elapsed_ms=float(fields.get('elapsed_ms', '0')),
        reason=_optional(fields.get('reason', 'None')),
    )


def shadow_records(lines: Iterable[str]) -> tuple[ShadowRecord, ...]:
    records = []
    for line in lines:
        # NOTE: production renders structlog as JSON with the message under event;
        # a console run prints it inline, so either form is read
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            message = line
        else:
            message = entry.get('event', '') if isinstance(entry, dict) else ''
        if (record := _record(message.rstrip())) is not None:
            records.append(record)
    return tuple(records)


def _agreement(records: list[ShadowRecord]) -> dict[str, object]:
    agreed = sum(record.choice == record.primary for record in records)
    return {
        'count': len(records),
        'agreed': agreed,
        'agreement': agreed / len(records) if records else None,
    }


def shadow_report(records: tuple[ShadowRecord, ...]) -> dict[str, object]:
    answered = [record for record in records if record.reason is None]
    by_primary = {
        primary: [record for record in answered if record.primary == primary]
        for primary in sorted({record.primary for record in answered})
    }

    buckets = {}
    for low, high in _BUCKETS:
        inside = [
            record
            for record in answered
            if record.confidence is not None
            and low <= record.confidence
            and (record.confidence < high or high == 1.0)
        ]
        buckets[f'{low:.1f}-{high:.1f}'] = _agreement(inside)

    disagreements = Counter(
        f'{record.primary} -> {record.choice}'
        for record in answered
        if record.choice != record.primary
    )
    latencies = [record.elapsed_ms for record in records]
    return {
        'shadowed': len(records),
        'answered': _agreement(answered),
        'fallbacks': dict(
            Counter(record.reason for record in records if record.reason is not None),
        ),
        'by_primary': {
            primary: _agreement(group) for primary, group in by_primary.items()
        },
        'by_confidence': buckets,
        'disagreements': dict(disagreements.most_common(10)),
        'latency_ms': {
            'p50': percentile(latencies, 0.5),
            'p95': percentile(latencies, 0.95),
        },
    }
