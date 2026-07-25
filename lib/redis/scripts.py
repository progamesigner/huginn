from pathlib import Path


def _lua(name: str) -> str:
    resources = Path(__file__).parent / 'resources'
    return (resources / f'{name}.lua').read_text(encoding='utf-8')


CLAIM = _lua('claim')
DRAIN = _lua('drain')
EXTEND = _lua('extend')
RELEASE = _lua('release')
