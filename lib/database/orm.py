import uuid
from datetime import datetime

from pydantic import JsonValue
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from .types import GUID, Json, Timestamp

_NAMING_CONVENTION = {
    'pk': 'pk_%(table_name)s',
    'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
    'ix': 'ix_%(column_0_label)s',
    'uq': 'uq_%(table_name)s_%(column_0_name)s',
    'ck': 'ck_%(table_name)s_%(constraint_name)s',
}

_TYPE_ANNOTATION_MAP = {
    datetime: Timestamp,
    JsonValue: Json,
    uuid.UUID: GUID,
}


class DatabaseDeclarativeBase(DeclarativeBase):
    metadata = MetaData(naming_convention=_NAMING_CONVENTION)

    type_annotation_map = _TYPE_ANNOTATION_MAP
