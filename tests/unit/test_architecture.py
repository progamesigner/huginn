import ast
import asyncio
import inspect
from collections.abc import Iterator
from pathlib import Path

from app.core import exceptions

CROSS_FEATURE_SURFACE: dict[tuple[str, str], frozenset[str]] = {
    ('agents', 'tools'): frozenset({'config', 'providers', 'registry'}),
}
FEATURES_PACKAGE = 'app.features'


def _features_dir() -> Path:
    import app

    return Path(app.__file__).parent / 'features'


def _package_of(path: Path, root: Path) -> str:
    return '.'.join((FEATURES_PACKAGE, *path.relative_to(root).parts[:-1]))


def _import_targets(source: str, package: str) -> Iterator[str]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                segments = package.split('.')
                anchor = '.'.join(segments[: len(segments) - (node.level - 1)])
                yield f'{anchor}.{node.module}' if node.module else anchor
            elif node.module:
                yield node.module


def _cross_feature_violation(feature: str, target: str) -> str | None:
    prefix = f'{FEATURES_PACKAGE}.'
    if not target.startswith(prefix):
        return None

    imported, _, rest = target[len(prefix) :].partition('.')
    if imported == feature:
        return None

    if not rest:
        return f'{target}: cross-feature import must name a submodule, not the package root'

    slice_ = rest.split('.')[0]
    if slice_ not in CROSS_FEATURE_SURFACE.get((feature, imported), frozenset()):
        return f'{target}: {imported}.{slice_} is not an approved cross-feature surface'

    return None


def test_skeleton_imports() -> None:
    import app
    import lib

    assert app is not None
    assert lib is not None


async def test_event_loop() -> None:
    assert asyncio.get_running_loop() is not None


def test_core_exceptions_hosts_only_app_error_family() -> None:
    defined = [
        member
        for _, member in inspect.getmembers(exceptions, inspect.isclass)
        if member.__module__ == exceptions.__name__ and issubclass(member, Exception)
    ]

    assert defined
    assert all(issubclass(member, exceptions.AppError) for member in defined)


def test_feature_packages_have_empty_init() -> None:
    root = _features_dir()
    offenders = [
        str(path.relative_to(root))
        for path in sorted(root.rglob('__init__.py'))
        if ast.parse(path.read_text()).body
    ]

    assert not offenders, f'feature __init__.py must be empty: {offenders}'


def test_cross_feature_imports_stay_within_approved_surface() -> None:
    root = _features_dir()
    violations = []

    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            continue

        feature = relative.parts[0]
        package = _package_of(path, root)
        for target in _import_targets(path.read_text(), package):
            if message := _cross_feature_violation(feature, target):
                violations.append(f'{relative}: {message}')

    assert not violations, 'cross-feature import violations:\n' + '\n'.join(violations)


def test_cross_feature_surface_check_flags_violations() -> None:
    assert _cross_feature_violation('agents', 'app.features.tools.registry') is None
    assert _cross_feature_violation('agents', 'app.features.agents.config') is None
    assert _cross_feature_violation('agents', 'lib.llm') is None

    assert _cross_feature_violation('agents', 'app.features.tools') is not None
    assert (
        _cross_feature_violation('agents', 'app.features.tools.websearch') is not None
    )
    assert _cross_feature_violation('probes', 'app.features.tools.registry') is not None
