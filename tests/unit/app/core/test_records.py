from sqlalchemy import inspect

from app.core.records import AppRecord, TimestampRecordMixin
from lib.database import DatabaseDeclarativeBase


def test_app_record_base_is_an_abstract_seam() -> None:
    assert issubclass(AppRecord, DatabaseDeclarativeBase)
    assert AppRecord.__abstract__ is True
    assert AppRecord.metadata is DatabaseDeclarativeBase.metadata


def test_app_record_defines_the_shared_primary_key() -> None:
    class IdentifiedTestRecord(AppRecord):
        __tablename__ = 'identified_test_records'

    columns = inspect(IdentifiedTestRecord).columns

    assert set(columns.keys()) == {'id'}
    assert columns.id.primary_key


def test_timestamp_mixin_adds_app_controlled_columns() -> None:
    class TimestampedTestRecord(TimestampRecordMixin, AppRecord):
        __tablename__ = 'timestamped_test_records'

    columns = inspect(TimestampedTestRecord).columns

    assert set(columns.keys()) == {'id', 'created_at', 'updated_at'}
    assert columns.created_at.server_default is None
    assert columns.updated_at.server_default is None
    assert columns.updated_at.onupdate is None
    assert not columns.created_at.nullable
    assert not columns.updated_at.nullable
