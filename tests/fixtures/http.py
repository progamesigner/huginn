import pytest

from tests.helpers.http import MockHTTP


@pytest.fixture
def http() -> MockHTTP:
    return MockHTTP()
