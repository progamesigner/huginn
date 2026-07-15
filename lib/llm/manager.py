from .client import ProviderConfig, ProviderRegistry


class LLMManager:
    registry: ProviderRegistry

    def __init__(self, providers: tuple[ProviderConfig, ...]) -> None:
        self.registry = ProviderRegistry(providers)

    def provider_for(self, model: str) -> ProviderConfig | None:
        return self.registry.resolve(model)

    async def dispose(self) -> None:
        return
