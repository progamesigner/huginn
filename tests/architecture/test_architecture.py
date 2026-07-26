import ast
import asyncio
import inspect
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import timedelta
from importlib import import_module
from pathlib import Path

from sqlalchemy.orm import configure_mappers

from app.core import exceptions
from app.core.records import AppRecord

# Features sit in tiers, outermost first. A feature may only import from a strictly
# lower tier, which keeps the feature graph a DAG however many features get added; a
# feature absent from this list may not cross the feature boundary at all.
FEATURE_TIERS: tuple[frozenset[str], ...] = (
    frozenset({'channels', 'skills', 'tools'}),
    frozenset({'conversations'}),
    frozenset({'identities'}),
    frozenset({'mcp'}),
    frozenset({'agents'}),
)
# A feature's public face: what it declares and how it is wired, never how it works.
# There is no exception list; a feature that needs more than this needs a contract on
# the other side. Schemas are public because a cross-feature query has to name the
# other feature's tables; its repositories stay private.
SHARED_SLICES = frozenset(
    {
        'config',
        'contracts',
        'exceptions',
        'providers',
        'schemas',
        'types',
    },
)
ROOT_PACKAGES = ('app', 'bin', 'lib')
FEATURES_PACKAGE = 'app.features'
EXTS_PACKAGE = 'app.exts'
# An ext may expose a vendor's transport as well as its client: `endpoints` for what
# the vendor calls inbound, `gateway` for what we call outbound. Both talk to the app
# only through app.core.channels ports, so an ext still never imports a feature.
EXT_SLICE_NAMES = frozenset(
    {
        'client',
        'endpoints',
        'exceptions',
        'gateway',
        'types',
    },
)
BANNED_MODULE_NAMES = frozenset({'models', 'utils'})


def _features_dir() -> Path:
    import app

    return Path(app.__file__).parent / 'features'


def _exts_dir() -> Path:
    import app

    return Path(app.__file__).parent / 'exts'


def _source_dirs(package: str) -> set[Path]:
    module = import_module(package)

    assert module.__file__ is not None
    root = Path(module.__file__).parent
    return {path.parent for path in root.rglob('*.py')}


def _package_of(path: Path, root: Path, package: str) -> str:
    return '.'.join((package, *path.relative_to(root).parts[:-1]))


def _module_of(path: Path, root: Path, package: str) -> str:
    return '.'.join((package, *path.relative_to(root).with_suffix('').parts))


def _schema_modules() -> tuple[str, ...]:
    root = _features_dir()
    return tuple(
        _module_of(path, root, FEATURES_PACKAGE)
        for path in sorted(root.rglob('schemas/*.py'))
        if path.name != '__init__.py'
    )


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


def _tier_of(feature: str) -> int | None:
    return next(
        (index for index, tier in enumerate(FEATURE_TIERS) if feature in tier),
        None,
    )


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
    mine, theirs = _tier_of(feature), _tier_of(imported)
    if mine is None or theirs is None:
        return f'{target}: {feature} and {imported} must both appear in FEATURE_TIERS'

    if mine >= theirs:
        return (
            f'{target}: {feature} may not import {imported};'
            f' feature tiers run outermost first and imports must go strictly down'
        )

    if slice_ not in SHARED_SLICES:
        return (
            f'{target}: {imported}.{slice_} is an implementation slice;'
            f' import one of {sorted(SHARED_SLICES)} instead'
        )

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


# A vendor fact an ext already owns must be read off its spec, never retyped. The
# forbidden values are derived from the specs themselves, so a new vendor or field
# widens the check for free and there is nothing to keep in sync.
@dataclass(frozen=True)
class _Fact:
    vendor: str
    instance: str
    field: str
    value: object


def _facts() -> tuple[_Fact, ...]:
    collected = []
    for path in sorted(_exts_dir().glob('*/types.py')):
        vendor = path.parent.name
        module = import_module(f'{EXTS_PACKAGE}.{vendor}.types')
        collected.extend(
            _Fact(
                vendor=vendor,
                instance=name,
                field=field.name,
                value=getattr(spec, field.name),
            )
            # NOTE: a spec is a frozen dataclass instance; the classes themselves
            # answer is_dataclass too, hence the isinstance guard
            for name, spec in vars(module).items()
            if is_dataclass(spec) and not isinstance(spec, type)
            for field in fields(spec)
        )

    return tuple(collected)


def _imported_vendors(targets: Iterable[str]) -> frozenset[str]:
    prefix = f'{EXTS_PACKAGE}.'
    return frozenset(
        target[len(prefix) :].partition('.')[0]
        for target in targets
        if target.startswith(prefix)
    )


def _forbidden(facts: Sequence[_Fact], vendors: frozenset[str]) -> frozenset[object]:
    values = set()
    for fact in facts:
        if fact.vendor not in vendors:
            continue

        values.add(fact.value)
        if isinstance(fact.value, timedelta):
            # NOTE: timedelta(seconds=1.1) is a call; only the 1.1 is a literal
            values.add(fact.value.total_seconds())

    return frozenset(value for value in values if value is not None)


def _retyped_facts(
    source: str,
    package: str,
    facts: Sequence[_Fact],
) -> Iterator[str]:
    vendors = _imported_vendors(_import_targets(source, package))
    forbidden = _forbidden(facts, vendors)
    if not forbidden:
        return

    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Constant)
            # NOTE: isinstance(True, int) holds, so booleans would match any 0/1 fact
            and not isinstance(node.value, bool)
            and node.value in forbidden
        ):
            yield f'line {node.lineno}: {node.value!r} is a vendor fact; read the spec'


def _attribute_reads(source: str) -> Iterator[tuple[str, str]]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            yield node.value.id, node.attr


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


def test_cross_feature_imports_respect_the_feature_tiers() -> None:
    root = _features_dir()
    violations = []

    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            continue

        feature = relative.parts[0]
        package = _package_of(path, root, FEATURES_PACKAGE)
        violations.extend(
            f'{relative}: {message}'
            for target in _import_targets(path.read_text(), package)
            if (message := _cross_feature_violation(feature, target))
        )

    assert not violations, 'cross-feature import violations:\n' + '\n'.join(violations)


def test_cross_feature_tier_check_allows_the_public_face_downward() -> None:
    assert _cross_feature_violation('agents', 'app.features.agents.config') is None
    assert _cross_feature_violation('agents', 'lib.llm') is None
    assert _cross_feature_violation('channels', 'app.features.agents.config') is None
    assert _cross_feature_violation('skills', 'app.features.agents.contracts') is None
    assert _cross_feature_violation('tools', 'app.features.agents.types') is None
    assert _cross_feature_violation('mcp', 'app.features.agents.providers') is None
    assert _cross_feature_violation('skills', 'app.features.mcp.providers') is None
    # a cross-feature query names the other feature's tables
    turns = 'app.features.conversations.schemas.sqlalchemy'
    users = 'app.features.identities.schemas.sqlalchemy'

    assert _cross_feature_violation('channels', turns) is None
    assert _cross_feature_violation('conversations', users) is None


def test_cross_feature_tier_check_flags_violations() -> None:
    # the package root hides which slice is really being used
    assert _cross_feature_violation('tools', 'app.features.agents') is not None
    # upward: every capability sits above agents, so agents must not reach back up
    assert _cross_feature_violation('agents', 'app.features.tools.registry') is not None
    assert (
        _cross_feature_violation('agents', 'app.features.skills.repositories')
        is not None
    )
    # sideways: two adapters in the same tier must not know each other
    assert _cross_feature_violation('channels', 'app.features.skills.types') is not None
    # downward but too deep: services is an implementation slice, not vocabulary
    assert _cross_feature_violation('tools', 'app.features.agents.services') is not None
    # unplaced features are denied outright
    assert _cross_feature_violation('probes', 'app.features.agents.types') is not None
    # agents is the innermost tier: nothing sits below it, so it stays sealed
    users = 'app.features.identities.schemas.sqlalchemy'

    assert _cross_feature_violation('agents', users) is not None
    # schemas are public, the repositories that query them are not
    assert (
        _cross_feature_violation('channels', 'app.features.conversations.repositories')
        is not None
    )


def test_orm_mappers_resolve_once_every_schema_is_imported() -> None:
    # NOTE: relationship()/back_populates/ForeignKey resolve lazily, so a bad
    # reference or a record module nobody imports raises on the first production
    # query and nowhere else. configure_mappers() forces that resolution here; it
    # touches metadata only, so this needs no database.
    modules = _schema_modules()

    assert modules, 'no schema module found; the check would pass vacuously'

    for module in modules:
        import_module(module)

    assert AppRecord.metadata.tables, 'importing the schemas mapped no table'

    configure_mappers()


def test_every_package_carries_an_init_module() -> None:
    # NOTE: grimp does not walk namespace packages, so a directory without an
    # __init__.py silently shrinks what the layers contract checks rather than
    # failing it. In 2026-07 that hid 86 of app/'s 98 modules from import-linter.
    missing = [
        str(directory)
        for package in ROOT_PACKAGES
        for directory in sorted(_source_dirs(package))
        if not (directory / '__init__.py').exists()
    ]

    assert not missing, 'these packages are invisible to import-linter:\n' + '\n'.join(
        missing,
    )


def test_exts_never_import_a_feature() -> None:
    # NOTE: the layers contract is meant to cover this, but grimp does not walk
    # namespace packages, so it sees 12 of the 98 modules under app/ and would pass
    # this silently. Until every package carries an __init__.py, this is the check.
    root = _exts_dir()
    violations = [
        f'{path.relative_to(root)}: {target}'
        for path in sorted(root.rglob('*.py'))
        for target in _import_targets(
            path.read_text(),
            _package_of(path, root, EXTS_PACKAGE),
        )
        if target == FEATURES_PACKAGE or target.startswith(f'{FEATURES_PACKAGE}.')
    ]

    assert not violations, 'an ext must not import a feature:\n' + '\n'.join(
        violations,
    )


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
    # a vendor's own transport, inbound and outbound
    assert _unknown_ext_name(Path('telegram/endpoints.py')) is None
    assert _unknown_ext_name(Path('telegram/gateway.py')) is None
    # orchestration still has no home here, whatever it is called
    assert _unknown_ext_name(Path('felo/service.py')) is not None
    assert _unknown_ext_name(Path('telegram/actor.py')) is not None
    assert _unknown_ext_name(Path('telegram/providers.py')) is not None


def test_feature_files_do_not_retype_vendor_facts() -> None:
    root = _features_dir()
    facts = _facts()

    assert facts, 'no ext spec found; the check would pass vacuously'

    violations = [
        f'{path.relative_to(root)}: {message}'
        for path in sorted(root.rglob('*.py'))
        for message in _retyped_facts(
            path.read_text(),
            _package_of(path, root, FEATURES_PACKAGE),
            facts,
        )
    ]

    assert not violations, 'retyped vendor facts:\n' + '\n'.join(violations)


def test_every_vendor_fact_is_read() -> None:
    # NOTE: an ext owns its transport now, so a fact its own endpoints read is in use
    reads = {
        read
        for root in (_features_dir(), _exts_dir())
        for path in sorted(root.rglob('*.py'))
        for read in _attribute_reads(path.read_text())
    }
    unread = [
        f'{fact.vendor}.{fact.instance}.{fact.field}'
        for fact in _facts()
        if (fact.instance, fact.field) not in reads
    ]

    assert not unread, 'vendor facts nobody reads:\n' + '\n'.join(unread)


def test_vendor_fact_check_flags_retyped_values_only_where_the_vendor_is_known() -> (
    None
):
    facts = (
        _Fact(
            vendor='telegram',
            instance='WEBHOOK',
            field='signature_header',
            value='X-Vendor-Header',
        ),
        _Fact(
            vendor='telegram',
            instance='MESSAGE',
            field='edit_interval',
            value=timedelta(seconds=1.1),
        ),
    )
    package = f'{FEATURES_PACKAGE}.channels.telegram'
    imports = f'from {EXTS_PACKAGE}.telegram.types import MESSAGE, WEBHOOK\n'

    def flagged(body: str) -> list[str]:
        return list(_retyped_facts(imports + body, package, facts))

    assert flagged("value = 'X-Vendor-Header'\n")
    assert flagged('wait = timedelta(seconds=1.1)\n')
    assert not flagged('value = WEBHOOK.signature_header\n')
    # a value no spec of a named vendor claims is none of the check's business
    assert not flagged('width = 200\n')
    # known blind spot: the check polices retyping, not first-time misplacement
    assert not list(_retyped_facts("value = 'X-Vendor-Header'\n", package, facts))


def test_attribute_read_check_sees_spec_field_access() -> None:
    assert ('WEBHOOK', 'ack_status') in set(
        _attribute_reads('status = WEBHOOK.ack_status\n'),
    )
    assert not set(_attribute_reads('status = 200\n'))
