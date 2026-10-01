import ast
import asyncio
import inspect
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import timedelta
from importlib import import_module
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import configure_mappers

from app.core import exceptions
from lib.container import ProviderContract
from tests.helpers.schemas import mapped_tables

# Features sit in tiers, outermost first. A feature may only import from a strictly
# lower tier, which keeps the feature graph a DAG however many features get added; a
# feature absent from this list may not cross the feature boundary at all.
FEATURE_TIERS: tuple[frozenset[str], ...] = (
    frozenset({'channels', 'messages', 'skills', 'tools'}),
    frozenset({'auth'}),
    frozenset({'conversations'}),
    frozenset({'identities'}),
    frozenset({'mcp'}),
    frozenset({'agents'}),
    frozenset({'blobs'}),
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
CONTRACTS_PACKAGE = 'app.contracts'
# A feature touches an ext only from a repository or a source. A data-source ext is
# pulled: the repository owns the call and translates the answer into domain values. A
# transport ext is pushed: it implements an app.contracts contract and app/providers
# wires it in, so the feature is handed the result without ever naming the vendor.
EXT_CLIENT_HOSTS = frozenset({'repositories', 'sources'})
# Whichever direction it reaches, a feature never names a vendor's payload types. Those
# are the vendor's own wire shapes: an ext parses them at its edge and hands over the
# app's own vocabulary, so a payload type in a feature means the boundary leaked.
EXT_VENDOR_TYPES_SLICE = 'types'
# A vendor reader is tolerant by nature -- it takes what the vendor sends and ignores what
# it does not know -- so what it rejects is somebody's message, a search answer, a
# transcript of a voice note. Pydantic quotes the rejected input in its error text by
# default, which hands that content to every caller that logs a validation failure, and
# `logger.exception` is the ordinary way to report one. Hiding inputs makes the payload
# reach a log only where a caller deliberately puts it there.
TOLERANT_EXTRA = 'ignore'
HIDDEN_INPUT = 'hide_input_in_errors'
BANNED_MODULE_NAMES = frozenset({'models', 'utils'})


def _features_dir() -> Path:
    import app

    return Path(app.__file__).parent / 'features'


def _exts_dir() -> Path:
    import app

    return Path(app.__file__).parent / 'exts'


def _contracts_dir() -> Path:
    import app

    return Path(app.__file__).parent / 'contracts'


def _tests_dir() -> Path:
    return Path(__file__).parent.parent


def _source_root(package: str) -> Path:
    module = import_module(package)

    assert module.__file__ is not None
    return Path(module.__file__).parent


def _source_dirs(package: str) -> set[Path]:
    return {path.parent for path in _source_root(package).rglob('*.py')}


def _package_of(path: Path, root: Path, package: str) -> str:
    return '.'.join((package, *path.relative_to(root).parts[:-1]))


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


def _ext_client_reach(relative: Path, targets: Iterable[str]) -> Iterator[str]:
    if {*relative.parts[:-1], relative.stem} & EXT_CLIENT_HOSTS:
        return

    prefix = f'{EXTS_PACKAGE}.'
    for target in targets:
        if not target.startswith(prefix):
            continue

        yield (
            f'{target}: a feature reaches a vendor only from {sorted(EXT_CLIENT_HOSTS)};'
            f' a transport is wired to it at app/providers through an app.contracts'
            f' contract instead'
        )


def _vendor_types_reach(targets: Iterable[str]) -> Iterator[str]:
    prefix = f'{EXTS_PACKAGE}.'
    for target in targets:
        if not target.startswith(prefix):
            continue

        slice_ = target[len(prefix) :].partition('.')[2].split('.')[0]
        if slice_ == EXT_VENDOR_TYPES_SLICE:
            yield (
                f'{target}: a vendor payload type stops at the ext;'
                f' have the adapter hand over an app.contracts type instead'
            )


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


# A contract is the seam a lower layer uses to call back into a higher one, so it is
# the one place the layer rules cannot police by direction alone. These two checks are
# what keep it from becoming a dumping ground: a contract declares a boundary and knows
# nothing about the app on either side of it, which leaves no room for an
# implementation to hide here.
def _implemented(source: str) -> Iterator[str]:
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue

        body = [
            statement
            for statement in node.body
            # NOTE: `...` and a docstring are both bare Expr(Constant) nodes
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
            )
        ]
        if body:
            yield f'line {node.lineno}: {node.name} has a body; a contract only declares'


def _reaches_the_app(source: str, package: str) -> Iterator[str]:
    for target in _import_targets(source, package):
        if (target == 'app' or target.startswith('app.')) and not target.startswith(
            f'{CONTRACTS_PACKAGE}.',
        ):
            yield f'{target}: a contract must not import the app whose boundary it declares'


# A signature declares types; a function body lets ty infer them. An annotation on a
# local is either redundant with what the signature already says or a sign the value is
# being built in a shape ty cannot follow -- both are worth rewriting rather than
# annotating. Class bodies and module level are the opposite case: there the annotation
# IS the declaration, so only function bodies are checked.
def _annotated_locals(source: str) -> Iterator[str]:
    def visit(node: ast.AST, *, in_function: bool) -> Iterator[str]:
        for child in ast.iter_child_nodes(node):
            match child:
                case ast.AnnAssign(target=ast.Name(id=name)) if in_function:
                    yield f'line {child.lineno}: {name} is annotated; let ty infer it'
                case ast.FunctionDef() | ast.AsyncFunctionDef():
                    yield from visit(child, in_function=True)
                case ast.ClassDef():
                    yield from visit(child, in_function=False)
                case _:
                    yield from visit(child, in_function=in_function)

    yield from visit(ast.parse(source), in_function=False)


def _attribute_reads(source: str) -> Iterator[tuple[str, str]]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            yield node.value.id, node.attr


def _ext_modules() -> Iterator[str]:
    root = _exts_dir()
    for path in sorted(root.rglob('*.py')):
        stem = path.relative_to(root).with_suffix('')
        parts = stem.parts[:-1] if stem.name == '__init__' else stem.parts
        yield '.'.join((EXTS_PACKAGE, *parts))


def _leaks_rejected_input(model: type[BaseModel]) -> bool:
    config = model.model_config
    return config.get('extra') == TOLERANT_EXTRA and not config.get(HIDDEN_INPUT, False)


def _readers_of(module: str) -> Iterator[type[BaseModel]]:
    for member in vars(import_module(module)).values():
        if (
            isinstance(member, type)
            and issubclass(member, BaseModel)
            # NOTE: an imported model belongs to the module that declares it
            and member.__module__ == module
        ):
            yield member


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


def test_function_bodies_leave_local_types_to_inference() -> None:
    roots = (*(_source_root(package) for package in ROOT_PACKAGES), _tests_dir())
    offenders = [
        f'{path}: {message}'
        for root in roots
        for path in sorted(root.rglob('*.py'))
        for message in _annotated_locals(path.read_text(encoding='utf-8'))
    ]

    assert not offenders, 'annotate the signature, not the local:\n' + '\n'.join(
        offenders,
    )


def test_annotated_local_check_reads_the_scope_not_the_syntax() -> None:
    def flagged(source: str) -> bool:
        return bool(list(_annotated_locals(source)))

    assert not flagged('value: int = 1\n')
    assert not flagged('class A:\n    value: int = 1\n')
    assert not flagged('def f() -> None:\n    class A:\n        value: int = 1\n')
    assert not flagged('def f(value: int = 1) -> None:\n    pass\n')

    assert flagged('def f() -> None:\n    value: int = 1\n')
    assert flagged('async def f() -> None:\n    value: list[int] = []\n')
    assert flagged('class A:\n    def f(self) -> None:\n        value: int = 1\n')


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
    assert mapped_tables()

    configure_mappers()


def _providers() -> Iterator[type[ProviderContract]]:
    for root in ('app', 'bin'):
        base = _source_root(root)
        for path in sorted(base.rglob('*.py')):
            if 'providers' not in {*path.relative_to(base).parts[:-1], path.stem}:
                continue
            module = import_module(f'{_package_of(path, base, root)}.{path.stem}')
            yield from (
                value
                for value in vars(module).values()
                if inspect.isclass(value)
                and issubclass(value, ProviderContract)
                and value.__module__ == module.__name__
            )


def test_providers_take_no_construction_arguments() -> None:
    # NOTE: options come from settings, so a provider is wired the same wherever it
    # is pulled in transitively and nothing has to list it just to pass an argument
    providers = list(_providers())
    violations = [
        provider.__qualname__
        for provider in providers
        if inspect.signature(provider).parameters
    ]

    assert providers, 'no provider found; the check would pass vacuously'
    assert not violations, 'providers taking arguments:\n' + '\n'.join(violations)


def _contribution_calls(source: str) -> Iterator[int]:
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'contribute'
        ):
            yield node.lineno


def test_only_the_composition_root_contributes_to_a_registry() -> None:
    # NOTE: a feature filling another's registry is an import of that feature in
    # disguise; the root sees every side, so it is the one place that may do it
    root = _source_root('app')
    violations = [
        f'{path.relative_to(root)}:{line}'
        for path in sorted(root.rglob('*.py'))
        if path.relative_to(root).parts[0] != 'providers'
        for line in _contribution_calls(path.read_text())
    ]

    assert not violations, 'contributions outside app/providers:\n' + '\n'.join(
        violations,
    )


def test_contribution_check_sees_attribute_calls() -> None:
    assert list(_contribution_calls('container.contribute(X, f)')) == [1]
    assert list(_contribution_calls('contribute = 1')) == []


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


def test_contracts_only_declare_their_boundary() -> None:
    root = _contracts_dir()
    modules = sorted(path for path in root.rglob('*.py') if path.name != '__init__.py')

    assert modules, 'no contract module found; the check would pass vacuously'

    violations = [
        f'{path.relative_to(root)}: {message}'
        for path in modules
        for message in _implemented(path.read_text())
    ]

    assert not violations, 'contracts must not implement anything:\n' + '\n'.join(
        violations,
    )


def test_contracts_never_reach_back_into_the_app() -> None:
    root = _contracts_dir()
    violations = [
        f'{path.relative_to(root)}: {message}'
        for path in sorted(root.rglob('*.py'))
        for message in _reaches_the_app(
            path.read_text(),
            _package_of(path, root, CONTRACTS_PACKAGE),
        )
    ]

    assert not violations, 'contracts must stay ignorant of the app:\n' + '\n'.join(
        violations,
    )


def test_contract_declaration_check_flags_a_body() -> None:
    assert not list(_implemented('class Boundary:\n    def read(self) -> int: ...\n'))
    assert not list(_implemented('def read() -> None:\n    """Doc."""\n'))
    assert not list(_implemented('MAX = 3\n'))

    assert list(_implemented('def read() -> int:\n    return 1\n'))
    assert list(
        _implemented('class Boundary:\n    def read(self) -> None:\n        pass\n'),
    )


def test_contract_import_check_flags_app_imports() -> None:
    assert not list(
        _reaches_the_app('from lib.router import Router\n', CONTRACTS_PACKAGE),
    )
    assert not list(
        _reaches_the_app('from dataclasses import dataclass\n', CONTRACTS_PACKAGE),
    )
    # a contract may lean on a sibling contract's vocabulary
    assert not list(
        _reaches_the_app('from .channels import InboundMessage\n', CONTRACTS_PACKAGE),
    )

    assert list(
        _reaches_the_app(
            'from app.features.agents.types import Tool\n',
            CONTRACTS_PACKAGE,
        ),
    )
    assert list(
        _reaches_the_app('from app.core.settings import Settings\n', CONTRACTS_PACKAGE),
    )


def test_feature_and_ext_files_avoid_banned_names() -> None:
    violations = [
        message
        for root in (_features_dir(), _exts_dir())
        for path in sorted(root.rglob('*.py'))
        if (message := _banned_name(path.relative_to(root)))
    ]

    assert not violations, 'banned file names:\n' + '\n'.join(violations)


def test_features_reach_an_ext_only_from_a_repository() -> None:
    root = _features_dir()
    violations = []

    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        targets = _import_targets(
            path.read_text(),
            _package_of(path, root, FEATURES_PACKAGE),
        )
        violations.extend(
            f'{relative}: {message}' for message in _ext_client_reach(relative, targets)
        )

    assert not violations, 'ext reached outside a repository:\n' + '\n'.join(
        violations,
    )


def test_features_never_name_a_vendor_payload_type() -> None:
    root = _features_dir()
    violations = [
        f'{path.relative_to(root)}: {message}'
        for path in sorted(root.rglob('*.py'))
        for message in _vendor_types_reach(
            _import_targets(
                path.read_text(),
                _package_of(path, root, FEATURES_PACKAGE),
            ),
        )
    ]

    assert not violations, 'vendor payload types inside a feature:\n' + '\n'.join(
        violations,
    )


def test_vendor_types_check_flags_only_the_vendor_payload_slice() -> None:
    assert not list(_vendor_types_reach(('app.exts.telegram.gateways',)))
    assert not list(_vendor_types_reach(('app.exts.felo.clients',)))
    assert not list(_vendor_types_reach(('app.contracts.channels',)))
    # our own types slice is a feature's own business
    assert not list(_vendor_types_reach(('app.features.agents.types',)))

    assert list(_vendor_types_reach(('app.exts.telegram.types',)))


def test_ext_reach_check_allows_only_repositories_and_sources() -> None:
    client = ('app.exts.felo.clients',)
    transport = ('app.exts.telegram.gateways', 'app.exts.telegram.endpoints')

    # a data source is a repository's to call, whichever shape repositories takes
    assert not list(_ext_client_reach(Path('tools/websearch/repositories.py'), client))
    assert not list(
        _ext_client_reach(Path('conversations/repositories/repository.py'), client),
    )
    assert not list(_ext_client_reach(Path('skills/sources/muninn.py'), client))
    # a transport is wired at the root, so not even the feature's own wiring names it
    assert list(_ext_client_reach(Path('channels/telegram/service.py'), transport))
    assert list(_ext_client_reach(Path('channels/telegram/providers.py'), transport))
    # our own vocabulary is nobody's business here
    assert not list(_ext_client_reach(Path('tools/websearch/tool.py'), ('lib.http',)))

    assert list(_ext_client_reach(Path('tools/websearch/tool.py'), client))
    assert list(
        _ext_client_reach(
            Path('channels/telegram/service.py'),
            ('app.exts.telegram.clients',),
        ),
    )


def test_vendor_readers_keep_rejected_input_out_of_their_errors() -> None:
    readers = [
        (module, reader) for module in _ext_modules() for reader in _readers_of(module)
    ]

    assert readers, 'no vendor reader found; the check would pass vacuously'

    violations = [
        f'{module}.{reader.__name__}'
        for module, reader in readers
        if _leaks_rejected_input(reader)
    ]

    assert not violations, (
        'these readers quote the rejected payload in their error text:\n'
        + '\n'.join(violations)
    )


def test_banned_name_check_flags_models_and_utils() -> None:
    assert _banned_name(Path('tools/repositories.py')) is None
    assert _banned_name(Path('tools/models.py')) is not None
    assert _banned_name(Path('tools/utils/helpers.py')) is not None


def test_rejected_input_check_flags_only_a_tolerant_reader() -> None:
    class Reader(BaseModel):
        model_config = ConfigDict(extra='ignore')

    class Hiding(BaseModel):
        model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)

    # our own strict models answer to the type rules, not to this one
    class Strict(BaseModel):
        model_config = ConfigDict(extra='forbid')

    assert _leaks_rejected_input(Reader)
    assert not _leaks_rejected_input(Hiding)
    assert not _leaks_rejected_input(Strict)


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
