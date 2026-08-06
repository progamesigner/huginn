import logging
from hmac import compare_digest
from typing import override

from app.contracts.auth import ApiClient, ApiRouting, AuthContract
from app.core.config.registry import HuginnConfig

from .config import API_KEYS, ApiKeyConfig

logger = logging.getLogger(__name__)

type Registered = tuple[tuple[bytes, ApiClient], ...]


def _client(name: str, key: ApiKeyConfig) -> ApiClient:
    return ApiClient(
        key=name,
        agents=frozenset(key.agents),
        agent=key.agent,
        routing=(
            None
            if key.router is None
            else ApiRouting(
                fallback=key.router.fallback,
                model=key.router.model,
                provider=key.router.provider,
            )
        ),
        denied_tools=frozenset(() if key.tools is None else key.tools.deny),
    )


def _registered(config: HuginnConfig) -> Registered:
    return tuple(
        (key.token.get_secret_value().encode(), _client(name, key))
        for name, key in config.section(API_KEYS).root.items()
    )


class ApiKeyDirectory(AuthContract):
    clients: Registered

    def __init__(self, config: HuginnConfig) -> None:
        self.clients = _registered(config)

    @override
    def authenticate(self, credential: str) -> ApiClient | None:
        offered = credential.encode()
        found = None

        for token, client in self.clients:
            if compare_digest(token, offered):
                found = client

        if found is None:
            logger.info('Rejected an unknown api key')

        return found
