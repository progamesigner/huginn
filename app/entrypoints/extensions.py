from app.features.agents.providers import AgentsConfigProvider, WebSearchToolProvider
from lib.container import ProviderContract

EXTENSIONS: tuple[ProviderContract, ...] = (
    AgentsConfigProvider(),
    WebSearchToolProvider(),
)
