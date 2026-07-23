from datetime import UTC, datetime, timedelta, timezone

import pytest

from lib.temporal import (
    EpochUnit,
    as_utc,
    assume_utc,
    from_epoch,
    isoformat,
    parse,
    parse_formats,
    to_epoch,
    truncate,
    with_timezone,
)

_PLUS_TWO = timezone(timedelta(hours=2))


def test_to_utc_converts_aware_to_utc() -> None:
    aware = datetime(2026, 7, 23, 12, tzinfo=_PLUS_TWO)

    assert as_utc(aware) == datetime(2026, 7, 23, 10, tzinfo=UTC)


def test_to_utc_rejects_naive() -> None:
    with pytest.raises(ValueError, match='naive'):
        as_utc(datetime.fromisoformat('2026-07-23T12:00:00'))


def test_assume_utc_labels_a_naive_datetime() -> None:
    assert assume_utc(datetime.fromisoformat('2026-07-23T12:00:00')) == datetime(
        2026, 7, 23, 12, tzinfo=UTC
    )


def test_assume_utc_rejects_aware() -> None:
    with pytest.raises(ValueError, match='naive'):
        assume_utc(datetime(2026, 7, 23, 12, tzinfo=UTC))


def test_to_zone_projects_to_a_display_zone() -> None:
    projected = with_timezone(datetime(2026, 7, 23, 10, tzinfo=UTC), _PLUS_TWO)

    assert projected.hour == 12
    assert projected.utcoffset() == timedelta(hours=2)


def test_to_zone_rejects_naive() -> None:
    with pytest.raises(ValueError, match='naive'):
        with_timezone(datetime.fromisoformat('2026-07-23T10:00:00'), _PLUS_TWO)


@pytest.mark.parametrize(
    'raw',
    ['2026-07-23T12:00:00+00:00', '2026-07-23T12:00:00Z'],
)
def test_parse_iso_returns_utc(raw: str) -> None:
    assert parse(raw) == datetime(2026, 7, 23, 12, tzinfo=UTC)


def test_parse_keeps_the_original_offset() -> None:
    parsed = parse('2026-07-23T12:00:00+02:00')

    assert parsed.utcoffset() == timedelta(hours=2)
    assert parsed == datetime(2026, 7, 23, 10, tzinfo=UTC)


def test_parse_naive_assumes_utc() -> None:
    parsed = parse('2026-07-23T12:00:00')

    assert parsed.tzinfo is UTC
    assert parsed == datetime(2026, 7, 23, 12, tzinfo=UTC)


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse('not-a-timestamp')


def test_parse_formats_tries_each_format_in_order() -> None:
    assert parse_formats('2026-07-23', '%Y-%m-%d', '%Y/%m/%d') == datetime(
        2026, 7, 23, tzinfo=UTC
    )
    assert parse_formats('2026/07/23', '%Y-%m-%d', '%Y/%m/%d') == datetime(
        2026, 7, 23, tzinfo=UTC
    )


def test_parse_formats_result_is_utc() -> None:
    parsed = parse_formats('2026-07-23 12:30', '%Y-%m-%d %H:%M')

    assert parsed.tzinfo is UTC
    assert parsed == datetime(2026, 7, 23, 12, 30, tzinfo=UTC)


def test_parse_formats_assumes_the_given_timezone() -> None:
    parsed = parse_formats('2026-07-23 12:00', '%Y-%m-%d %H:%M', tz=_PLUS_TWO)

    assert parsed.utcoffset() == timedelta(hours=2)
    assert parsed == datetime(2026, 7, 23, 10, tzinfo=UTC)


def test_parse_formats_guesses_common_formats_when_none_given() -> None:
    assert parse_formats('2026/07/23') == datetime(2026, 7, 23, tzinfo=UTC)
    assert parse_formats('2026-07-23 12:30:45') == datetime(
        2026, 7, 23, 12, 30, 45, tzinfo=UTC
    )


def test_parse_formats_raises_when_nothing_matches() -> None:
    with pytest.raises(ValueError, match='candidate formats'):
        parse_formats('not-a-date', '%Y-%m-%d', '%Y/%m/%d')


@pytest.mark.parametrize('unit', ['s', 'ms', 'us'])
def test_epoch_round_trips_at_unit_granularity(unit: EpochUnit) -> None:
    moment = datetime(2026, 7, 23, 12, 30, 45, tzinfo=UTC)

    assert from_epoch(to_epoch(moment, unit=unit), unit=unit) == moment


def test_epoch_preserves_microsecond_precision() -> None:
    moment = datetime(2026, 7, 23, 12, 30, 45, 123456, tzinfo=UTC)

    assert (
        to_epoch(moment, unit='us') == to_epoch(moment, unit='s') * 1_000_000 + 123456
    )
    assert from_epoch(to_epoch(moment, unit='us'), unit='us') == moment


def test_epoch_zero_is_the_unix_epoch() -> None:
    assert from_epoch(0) == datetime(1970, 1, 1, tzinfo=UTC)
    assert to_epoch(datetime(1970, 1, 1, tzinfo=UTC)) == 0


def test_isoformat_is_canonical_utc() -> None:
    aware = datetime(2026, 7, 23, 12, tzinfo=_PLUS_TWO)

    assert isoformat(aware) == '2026-07-23T10:00:00+00:00'


def test_truncate_to_day() -> None:
    moment = datetime(2026, 7, 23, 12, 30, 45, 123456, tzinfo=UTC)

    assert truncate(moment, 'day') == datetime(2026, 7, 23, tzinfo=UTC)


def test_truncate_to_hour() -> None:
    moment = datetime(2026, 7, 23, 12, 30, 45, tzinfo=UTC)

    assert truncate(moment, 'hour') == datetime(2026, 7, 23, 12, tzinfo=UTC)


def test_truncate_preserves_the_original_zone() -> None:
    truncated = truncate(datetime(2026, 7, 23, 12, 30, tzinfo=_PLUS_TWO), 'day')

    assert truncated == datetime(2026, 7, 23, tzinfo=_PLUS_TWO)
    assert truncated.utcoffset() == timedelta(hours=2)


def test_truncate_rejects_naive() -> None:
    with pytest.raises(ValueError, match='naive'):
        truncate(datetime.fromisoformat('2026-07-23T12:00:00'), 'day')
