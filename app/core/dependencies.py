from dataclasses import dataclass
from typing import cast

from app.contracts.auth import ApiClient, AuthContract
from lib.container import Container
from lib.router import Request

from .exceptions import UnauthorizedError

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

        return client
