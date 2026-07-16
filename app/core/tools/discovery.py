import importlib
import pkgutil
from typing import cast

from .base import Tool, ToolParams
from .registry import ToolRegistry

type Packages = tuple[str, ...]

DEFAULT_PACKAGES = cast(Packages, ('app.exts', 'app.features'))


def discover_tools(
    packages: Packages = DEFAULT_PACKAGES,
) -> tuple[Tool[ToolParams], ...]:
    tools: list[Tool[ToolParams]] = []

    for package_name in packages:
        package = importlib.import_module(package_name)

        for module_info in pkgutil.walk_packages(
            package.__path__,
            prefix=f'{package_name}.',
        ):
            if not module_info.name.endswith('.tools'):
                continue

            module = importlib.import_module(module_info.name)
            tools.extend(getattr(module, 'TOOLS', ()))

    return tuple(tools)


def build_registry(packages: Packages = DEFAULT_PACKAGES) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in discover_tools(packages):
        registry.register(tool)
    return registry
