import pytest

from app.features.identities.validation import registration_handle


@pytest.mark.parametrize(
    'value',
    [
        '.',
        '..',
        '../person',
        'a/b',
        'a\\b',
        'a:b',
        '',
        'a' * 65,
        'default',
        'shared',
        'admin',
        'system',
        'root',
        'anonymous',
    ],
)
def test_registration_rejects_unsafe_and_reserved_names(value: str) -> None:
    with pytest.raises(ValueError, match=r'handle|reserved|scope'):
        registration_handle(value)


@pytest.mark.parametrize(
    'value',
    ['a', 'person.name', 'person-name', 'person_name', 'a' * 64],
)
def test_registration_accepts_safe_readable_handles(value: str) -> None:
    assert registration_handle(value) == value
