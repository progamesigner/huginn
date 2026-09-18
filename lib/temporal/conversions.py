import re
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Literal

type EpochUnit = Literal['s', 'ms', 'us']
type TruncateUnit = Literal['day', 'hour', 'minute', 'second']

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_DURATION = re.compile(r'(?P<amount>[1-9][0-9]*)(?P<unit>[smhd])')
_DURATION_UNITS = {
    's': timedelta(seconds=1),
    'm': timedelta(minutes=1),
    'h': timedelta(hours=1),
    'd': timedelta(days=1),
}
_MICROS_PER_UNIT = {'s': 1_000_000, 'ms': 1_000, 'us': 1}
_COMMON_FORMATS = (
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
    '%Y-%m-%d',
    '%Y/%m/%d %H:%M:%S',
    '%Y/%m/%d %H:%M',
    '%Y/%m/%d',
    '%Y.%m.%d',
)


def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError(
            'naive datetime is not allowed; use assume_utc for known-UTC values',
        )
    return dt.astimezone(UTC)


def assume_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        raise ValueError(
            'assume_utc expects a naive datetime; use as_utc for aware values',
        )
    return dt.replace(tzinfo=UTC)


def parse(s: str) -> datetime:
    parsed = datetime.fromisoformat(s)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def parse_formats(s: str, *formats: str, tz: tzinfo = UTC) -> datetime:
    for fmt in formats or _COMMON_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=tz)
        except ValueError:
            continue
    raise ValueError(f'{s!r} matched none of the candidate formats')


def from_epoch(value: int, *, unit: EpochUnit = 's') -> datetime:
    return _EPOCH + timedelta(microseconds=value * _MICROS_PER_UNIT[unit])


def to_epoch(dt: datetime, *, unit: EpochUnit = 's') -> int:
    micros = (as_utc(dt) - _EPOCH) // timedelta(microseconds=1)
    return micros // _MICROS_PER_UNIT[unit]


def isoformat(dt: datetime) -> str:
    return as_utc(dt).isoformat()


def truncate(dt: datetime, unit: TruncateUnit) -> datetime:
    if dt.tzinfo is None:
        raise ValueError('naive datetime is not allowed')
    match unit:
        case 'day':
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        case 'hour':
            return dt.replace(minute=0, second=0, microsecond=0)
        case 'minute':
            return dt.replace(second=0, microsecond=0)
        case 'second':
            return dt.replace(microsecond=0)


def with_timezone(dt: datetime, tz: tzinfo) -> datetime:
    return as_utc(dt).astimezone(tz)


def parse_duration(value: str) -> timedelta:
    match = _DURATION.fullmatch(value)
    if match is None:
        raise ValueError(f'{value!r} is not a positive duration such as 10m, 1h or 1d')
    return int(match['amount']) * _DURATION_UNITS[match['unit']]
