import ast
import asyncio
import inspect
from collections.abc import Iterator
from pathlib import Path

from app.core import exceptions

CROSS_FEATURE_SURFACE: dict[tuple[str, str], frozenset[str]] = {
    ('agents', 'tools'): frozenset(
        {
            'binding',
            'config',
            'mcp',
            'providers',
            'registry',
        },
    ),
    ('agents', 'skills'): frozenset({'repositories', 'tools'}),
    ('skills', 'tools'): frozenset({'providers', 'types'}),
    ('channels', 'agents'): frozenset({'config', 'exceptions'}),
    ('agents', 'channels'): frozenset({'config'}),
}
FEATURES_PACKAGE = 'app.features'
EXT_SLICE_NAMES = frozenset({'client', 'types', 'exceptions'})
BANNED_MODULE_NAMES = frozenset({'models', 'utils'})


def _features_dir() -> Path:
    import app

    return Path(app.__file__).parent / 'features'


def _exts_dir() -> Path:
    import app

    return Path(app.__file__).parent / 'exts'


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


def _init_reexport_only(source: str) -> bool:
    # NOTE: slice subpackages may re-export their public face; nothing else in __init__
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.level >= 1:
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == '__all__'
            for target in node.targets
        ):
            continue
        return False

    return True


def _banned_name(relative: Path) -> str | None:
    if banned := {*relative.parts[:-1], relative.stem} & BANNED_MODULE_NAMES:
        return f'{relative}: banned file name {sorted(banned)}'

    return None


def _unknown_ext_name(relative: Path) -> str | None:
    if relative.name == '__init__.py' or relative.stem in EXT_SLICE_NAMES:
        return None

    return f'{relative}: unknown ext file name (allowed: {sorted(EXT_SLICE_NAMES)})'


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


def test_feature_package_inits_are_empty_or_relative_reexports() -> None:
    root = _features_dir()
    offenders = [
        str(path.relative_to(root))
        for path in sorted(root.rglob('__init__.py'))
        if not _init_reexport_only(path.read_text())
    ]

    assert not offenders, (
        f'feature __init__.py must be empty or relative re-exports only: {offenders}'
    )


def test_init_reexport_check_flags_logic_and_absolute_imports() -> None:
    assert _init_reexport_only('')
    assert _init_reexport_only(
        'from .repository import ConversationRepository\n\n'
        "__all__ = ['ConversationRepository']\n",
    )

    assert not _init_reexport_only('import os\n')
    assert not _init_reexport_only('from app.core import exceptions\n')
    assert not _init_reexport_only('value = compute()\n')


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


def test_feature_and_ext_files_avoid_banned_names() -> None:
    violations = [
        message
        for root in (_features_dir(), _exts_dir())
        for path in sorted(root.rglob('*.py'))
        if (message := _banned_name(path.relative_to(root)))
    ]

    assert not violations, 'banned file names:\n' + '\n'.join(violations)


def test_ext_files_use_ext_vocabulary() -> None:
    root = _exts_dir()
    violations = [
        message
        for path in sorted(root.rglob('*.py'))
        if (message := _unknown_ext_name(path.relative_to(root)))
    ]

    assert not violations, 'unknown ext file names:\n' + '\n'.join(violations)


def test_banned_name_check_flags_models_and_utils() -> None:
    assert _banned_name(Path('tools/repositories.py')) is None
    assert _banned_name(Path('tools/models.py')) is not None
    assert _banned_name(Path('tools/utils/helpers.py')) is not None


def test_ext_vocabulary_check_flags_unknown_names() -> None:
    assert _unknown_ext_name(Path('felo/client.py')) is None
    assert _unknown_ext_name(Path('felo/__init__.py')) is None
    assert _unknown_ext_name(Path('felo/service.py')) is not None
