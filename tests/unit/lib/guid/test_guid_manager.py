from lib.guid import GUIDManager


def test_generate_returns_uuid_version_7() -> None:
    assert GUIDManager().generate().version == 7


def test_generate_is_monotonic_within_a_process() -> None:
    manager = GUIDManager()
    ids = [manager.generate() for _ in range(100)]

    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
