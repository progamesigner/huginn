from typing import override

from structlog.stdlib import BoundLogger

from .adapters.base import HTTPDecisionAdapter
from .adapters.bifrost import BifrostAdapter
from .adapters.gemini import GeminiAdapter
from .adapters.openai import OpenAIAdapter
from .adapters.typesafe import TypeSafeAdapter
from .client import DecisionClient, DecisionProviderConfig, DecisionProviderRegistry
from .contracts import DecisionContract, HTTPConnector
from .types import DecisionRef


def build_adapter(
    provider: DecisionProviderConfig,
    http: HTTPConnector,
) -> HTTPDecisionAdapter:
    match provider.adapter:
        case 'gemini':
            return GeminiAdapter(provider, http)
        case 'bifrost':
            return BifrostAdapter(provider, http)
        case 'openai':
            return OpenAIAdapter(provider, http)
        case 'typesafe':
            return TypeSafeAdapter(provider, http)


class DecisionManager(DecisionContract):
    clients: dict[str, DecisionClient]
    http: HTTPConnector
    registry: DecisionProviderRegistry

    def __init__(
        self,
        providers: tuple[DecisionProviderConfig, ...],
        http: HTTPConnector,
        logger: BoundLogger,
    ) -> None:
        self.registry = DecisionProviderRegistry(providers, logger)
        self.http = http
        self.clients = {}

    @override
    def client_for(self, ref: DecisionRef) -> DecisionClient | None:
        provider = self.provider_for(ref)
        if provider is None:
            return None

        if provider.name not in self.clients:
            self.clients[provider.name] = build_adapter(provider, self.http)

        return self.clients[provider.name]

    @override
    def provider_for(self, ref: DecisionRef) -> DecisionProviderConfig | None:
        return self.registry.resolve(ref)
