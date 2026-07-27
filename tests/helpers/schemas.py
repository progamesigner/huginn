from importlib import import_module
from pathlib import Path

from app.core.records import AppRecord

FEATURES_PACKAGE = 'app.features'


def schema_modules() -> tuple[str, ...]:
    import app

    root = Path(app.__file__).parent / 'features'
    return tuple(
        '.'.join((FEATURES_PACKAGE, *path.relative_to(root).with_suffix('').parts))
        for path in sorted(root.rglob('schemas/*.py'))
        if path.name != '__init__.py'
    )


# NOTE: a table only reaches the metadata once its schema module is imported, so every
# caller that reads the metadata has to import first or it silently checks a subset
def mapped_tables() -> frozenset[str]:
    modules = schema_modules()

    assert modules, 'no schema module found; the caller would check nothing'

    for module in modules:
        import_module(module)

    assert AppRecord.metadata.tables, 'importing the schemas mapped no table'
    return frozenset(AppRecord.metadata.tables)
