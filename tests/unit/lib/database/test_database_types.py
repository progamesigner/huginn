import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import JSON, LargeBinary
from sqlalchemy.dialects import postgresql, sqlite

from lib.database import GUID, Json, Timestamp
from lib.database.orm import DatabaseDeclarativeBase

_PG = postgresql.dialect()
_SQLITE = sqlite.dialect()


def test_guid_uses_native_uuid_on_postgresql() -> None:
    impl = GUID().load_dialect_impl(_PG)
    assert isinstance(impl, postgresql.UUID)


def test_guid_falls_back_to_binary_16_on_other_dialects() -> None:
    impl = GUID().load_dialect_impl(_SQLITE)
    assert isinstance(impl, LargeBinary)
    assert impl.length == 16


def test_guid_round_trips_as_uuid_on_postgresql() -> None:
    value = uuid.uuid7()
    guid = GUID()

    assert guid.process_bind_param(value, _PG) is value
    assert guid.process_result_value(value, _PG) == value


def test_guid_round_trips_through_bytes_on_fallback() -> None:
    value = uuid.uuid7()
    guid = GUID()

    stored = guid.process_bind_param(value, _SQLITE)
    assert stored == value.bytes
    assert guid.process_result_value(stored, _SQLITE) == value


def test_guid_accepts_str_bind_value() -> None:
    value = uuid.uuid7()
    guid = GUID()

    assert guid.process_bind_param(str(value), _PG) == value
    assert guid.process_bind_param(str(value), _SQLITE) == value.bytes


def test_guid_accepts_bytes_bind_value() -> None:
    value = uuid.uuid7()
    guid = GUID()

    assert guid.process_bind_param(value.bytes, _PG) == value
    assert guid.process_bind_param(value.bytes, _SQLITE) == value.bytes


def test_json_uses_jsonb_on_postgresql_and_json_elsewhere() -> None:
    assert isinstance(Json().load_dialect_impl(_PG), postgresql.JSONB)
    assert isinstance(Json().load_dialect_impl(_SQLITE), JSON)


def test_timestamp_normalises_aware_datetimes_to_utc() -> None:
    aware = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=8)))

    result = Timestamp().process_bind_param(aware, _PG)

    assert result is not None
    assert result.tzinfo is UTC
    assert result.hour == 4


def test_timestamp_rejects_naive_datetimes() -> None:
    naive = datetime(2026, 1, 1, 12, tzinfo=UTC).replace(tzinfo=None)

    with pytest.raises(ValueError, match='timezone-aware'):
        Timestamp().process_bind_param(naive, _PG)


def test_base_maps_python_types_to_house_column_types() -> None:
    mapping = DatabaseDeclarativeBase.type_annotation_map
    assert mapping[uuid.UUID] is GUID
    assert mapping[datetime] is Timestamp
