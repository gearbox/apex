"""Database-only SQLAlchemy value adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from sqlalchemy.dialects.postgresql import BIT
from sqlalchemy.types import TypeDecorator

if TYPE_CHECKING:
    from sqlalchemy.engine.interfaces import Dialect


class PdqBit256(TypeDecorator[bytes]):
    """Map canonical 32-byte PDQ values to PostgreSQL ``BIT(256)`` safely."""

    impl = BIT(256)
    cache_ok = True

    @override
    def process_bind_param(self, value: bytes | None, dialect: Dialect) -> bytes | None:
        if value is None:
            return None
        if not isinstance(value, bytes) or len(value) != 32:
            raise ValueError("PDQ database values must be exactly 32 bytes")
        # asyncpg's PostgreSQL BIT codec accepts the binary 32-byte payload,
        # not a Python ``str`` made of zero/one characters.
        return value

    @override
    def process_result_value(self, value: object | None, dialect: Dialect) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, bytes) and len(value) == 32:
            return value
        raw_bytes = getattr(value, "bytes", None)
        if isinstance(raw_bytes, bytes) and len(raw_bytes) == 32:
            return raw_bytes
        # asyncpg returns its ``BitString`` wrapper here; its string form is
        # accessed through ``as_string()``; other PostgreSQL drivers commonly
        # return that string directly.
        as_string = getattr(value, "as_string", None)
        if callable(as_string):
            value = as_string()
        bits = value.decode("ascii") if isinstance(value, bytes) else str(value)
        if len(bits) != 256 or set(bits) - {"0", "1"}:
            raise ValueError("database returned an invalid BIT(256) value")
        return int(bits, 2).to_bytes(32, "big")
