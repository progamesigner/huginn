from app.exts.felo.providers import FeloExtToolsProvider
from lib.container import ProviderContract

EXTENSIONS: tuple[ProviderContract, ...] = (FeloExtToolsProvider(),)
