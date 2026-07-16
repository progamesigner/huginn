import asyncio
import inspect

from app.core import exceptions


def test_skeleton_imports() -> None:
    import app
    import lib

    assert app is not None
    assert lib is not None


async def test_event_loop() -> None:
    assert asyncio.get_running_loop() is not None


def test_core_exceptions_hosts_only_app_error_family() -> None:
    defined = [
        member
        for _, member in inspect.getmembers(exceptions, inspect.isclass)
        if member.__module__ == exceptions.__name__ and issubclass(member, Exception)
    ]

    assert defined
    assert all(issubclass(member, exceptions.AppError) for member in defined)
