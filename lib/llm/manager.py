from .adapters.anthropic import AnthropicAdapter
from .adapters.openai import OpenAIAdapter
from .client import LLMClient, ProviderConfig, ProviderRegistry


def build_adapter(provider: ProviderConfig) -> LLMClient:
    match provider.adapter:
        case 'anthropic':
            return AnthropicAdapter(provider)
        case 'openai':
            return OpenAIAdapter(provider)


class LLMManager:
    clients: dict[str, LLMClient]
    registry: ProviderRegistry

    def __init__(self, providers: tuple[ProviderConfig, ...]) -> None:
        self.registry = ProviderRegistry(providers)
        self.clients = {}

    def client_for(self, model: str) -> LLMClient | None:
        provider = self.provider_for(model)
        if provider is None:
            return None

        if provider.name not in self.clients:
            self.clients[provider.name] = build_adapter(provider)

        return self.clients[provider.name]

    def provider_for(self, model: str) -> ProviderConfig | None:
        return self.registry.resolve(model)

    async def dispose(self) -> None:
        for client in self.clients.values():
            await client.dispose()
