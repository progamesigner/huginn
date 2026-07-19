from app.core.records import AppRecord
from lib.database import DatabaseDeclarativeBase


def test_app_record_base_is_an_abstract_seam() -> None:
    assert issubclass(AppRecord, DatabaseDeclarativeBase)
    assert AppRecord.__abstract__ is True
    assert AppRecord.metadata is DatabaseDeclarativeBase.metadata
