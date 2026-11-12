from collections.abc import Mapping

import yaml

from app.features.skills.entities import SkillMeta
from lib.observability import Logger


def _split(content: str) -> tuple[Mapping[str, object], str]:
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != '---':
        return {}, content
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == '---'), None)
    if end is None:
        raise ValueError('unterminated skill frontmatter')
    try:
        parsed = yaml.safe_load(''.join(lines[1:end]))
    except yaml.YAMLError as error:
        raise ValueError('invalid skill frontmatter') from error
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise TypeError('skill frontmatter must be a mapping')
    return parsed, ''.join(lines[end + 1 :]).lstrip('\r\n')


def _name_from_path(path: str) -> str:
    parts = [part for part in path.split('/') if part]
    return parts[-2] if len(parts) >= 2 else path


def _meta(path: str, content: str, *, logger: Logger) -> SkillMeta:
    front, _ = _split(content)
    name = str(front.get('name') or _name_from_path(path))
    description = front.get('description')
    if not description:
        logger.warning('skill %s has no description; using its name', path)
        description = name
    return SkillMeta(name=name, description=str(description), path=path)
