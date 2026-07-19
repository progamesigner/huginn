from typing import override
from uuid import UUID

from app.core.application import App
from app.core.providers.guid import GUIDProvider
from lib.guid import GUIDContract, GUIDManager


class FakeGUID(GUIDContract):
    def __init__(self) -> None:
        self.counter = 0

    @override
    def generate(self) -> UUID:
        self.counter += 1
        return UUID(int=self.counter)


def test_provider_binds_guid_manager() -> None:
    container = App().build(GUIDProvider())

    assert isinstance(container.resolve(GUIDContract), GUIDManager)


def test_fake_guid_swaps_in_via_container_use() -> None:
    container = App().build(GUIDProvider())

    with container.use(GUIDContract, FakeGUID()):
        first = container.resolve(GUIDContract).generate()
        second = container.resolve(GUIDContract).generate()

    assert (first, second) == (UUID(int=1), UUID(int=2))
