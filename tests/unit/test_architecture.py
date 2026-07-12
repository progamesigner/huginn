import asyncio


def test_skeleton_imports() -> None:
    import app
    import lib

    assert app is not None
    assert lib is not None


async def test_event_loop() -> None:
    assert asyncio.get_running_loop() is not None
