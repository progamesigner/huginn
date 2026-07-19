from datetime import UTC, datetime
from typing import Any, override
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import JSON, DateTime, Dialect, LargeBinary, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.types import TypeEngine


class GUID(TypeDecorator[UUID]):
    cache_ok = True

    # NOTE: PostgreSQL stores native uuid (16 bytes); other dialects fall back to bytes
    impl = LargeBinary(16)

    @override
    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if self._is_native_dialect(dialect):
            return dialect.type_descriptor(PgUUID(as_uuid=True))

        return dialect.type_descriptor(LargeBinary(16))

    @override
    def process_bind_param(
        self,
        value: str | bytes | UUID | None,
        dialect: Dialect,
    ) -> UUID | bytes | None:
        if value is None:
            return None

        if not isinstance(value, UUID):
            value = UUID(bytes=value) if isinstance(value, bytes) else UUID(value)

        if self._is_native_dialect(dialect):
            return value

        return value.bytes

    @override
    def process_result_value(
        self,
        value: bytes | str | UUID | None,
        dialect: Dialect,
    ) -> UUID | None:
        if value is None:
            return None

        if isinstance(value, UUID):
            return value

        return UUID(bytes=value) if isinstance(value, bytes) else UUID(value)

    def _is_native_dialect(self, dialect: Dialect) -> bool:
        return dialect.name == 'postgresql'


class Json(TypeDecorator[JsonValue]):
    impl = JSON
    cache_ok = True

    @override
    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if self._is_native_dialect(dialect):
            return dialect.type_descriptor(JSONB())

        return dialect.type_descriptor(JSON())

    def _is_native_dialect(self, dialect: Dialect) -> bool:
        return dialect.name == 'postgresql'


class Timestamp(TypeDecorator[datetime]):
    cache_ok = True

    impl = DateTime(timezone=True)

    @override
    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        if value is None:
            return None

        if value.tzinfo is None:
            raise ValueError('Timestamp requires a timezone-aware datetime')

        return value.astimezone(UTC)
