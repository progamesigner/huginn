from app.features.agents.providers import AgentsConfigProvider
from app.features.tools.providers import WebSearchToolProvider
from lib.container import ProviderContract

EXTENSIONS: tuple[ProviderContract, ...] = (
    AgentsConfigProvider(),
    WebSearchToolProvider(),
)
