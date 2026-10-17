from dataclasses import dataclass, replace
from typing import cast

from app.contracts.auth import ApiClient, AuthContract
from lib.container import Container
from lib.router import Request

from .exceptions import InvalidRequestError, UnauthorizedError

_SCHEME = 'bearer'
_CHALLENGE = {'WWW-Authenticate': 'Bearer'}


@dataclass(frozen=True)
class Provide[T]:
    binding: type[T]

    def __call__(self, request: Request) -> T:
        return cast(Container, request.app.state.container).resolve(self.binding)


@dataclass(frozen=True)
class Authenticated:
    def __call__(self, request: Request) -> ApiClient:
        header = request.headers.get('authorization') or ''
        scheme, _, credential = header.partition(' ')
        client = None
        if scheme.lower() == _SCHEME and credential.strip():
            client = Provide(AuthContract)(request).authenticate(credential.strip())

        if client is None:
            raise UnauthorizedError(
                'a valid bearer token is required',
                headers=_CHALLENGE,
            )

        asserted = request.headers.get('x-huginn-user')
        if asserted is not None:
            if (
                not asserted.strip()
                or len(asserted) > 256
                or any(ord(c) < 32 or ord(c) == 127 for c in asserted)
            ):
                raise InvalidRequestError(
                    'X-Huginn-User must contain 1 to 256 printable characters',
                )
            client = replace(client, asserted_user=asserted)
        return client
