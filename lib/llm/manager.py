from typing import override

from lib.http import HTTPClientContract

from .client import LLMClient, ProviderConfig, ProviderRegistry
from .contracts import LLMContract


def build_adapter(provider: ProviderConfig, http: HTTPClientContract) -> LLMClient:
    try:
        match provider.adapter:
            case 'anthropic':
                from .adapters.anthropic import AnthropicAdapter

                return AnthropicAdapter(provider, http)
            case 'openai':
                from .adapters.openai import OpenAIAdapter

                return OpenAIAdapter(provider, http)
    except ImportError as error:
        message = (
            f'provider {provider.name!r} needs the {provider.adapter!r} extra; '
            f'install huginn[{provider.adapter}]'
        )
        raise ModuleNotFoundError(message) from error


class LLMManager(LLMContract):
    clients: dict[str, LLMClient]
    http: HTTPClientContract
    registry: ProviderRegistry

    def __init__(
        self,
        providers: tuple[ProviderConfig, ...],
        http: HTTPClientContract,
    ) -> None:
        self.registry = ProviderRegistry(providers)
        self.http = http
        self.clients = {}

    @override
    def client_for(self, model: str) -> LLMClient | None:
        provider = self.provider_for(model)
        if provider is None:
            return None

        if provider.name not in self.clients:
            self.clients[provider.name] = build_adapter(provider, self.http)

        return self.clients[provider.name]

    @override
    def provider_for(self, model: str) -> ProviderConfig | None:
        return self.registry.resolve(model)
