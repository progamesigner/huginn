import uuid
from typing import override
from uuid import UUID

from .contracts import GUIDContract


class GUIDManager(GUIDContract):
    @override
    def generate(self) -> UUID:
        # NOTE: UUIDv7 is time-ordered and monotonic within a process
        return uuid.uuid7()
