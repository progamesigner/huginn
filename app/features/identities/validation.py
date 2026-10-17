import re

_HANDLE = re.compile(r'[a-z0-9._-]{1,64}')
_SCOPE = re.compile(r'[A-Za-z0-9._-]{1,64}')
_RESERVED = frozenset({'default', 'shared', 'admin', 'system', 'root', 'anonymous'})


def normalize_handle(value: str) -> str:
    normalized = value.strip().lower()
    if not _HANDLE.fullmatch(normalized) or normalized in {'.', '..'}:
        raise ValueError(
            'handle must contain 1 to 64 lowercase letters, digits, ., _ or -',
        )
    return normalized


def validate_scope(value: str) -> str:
    if not _SCOPE.fullmatch(value) or value in {'.', '..'}:
        raise ValueError('invalid memory scope')
    return value


def registration_handle(value: str) -> str:
    handle = normalize_handle(value)
    if handle in _RESERVED:
        raise ValueError('that name is reserved')
    validate_scope(handle)
    return handle
