from collections.abc import Mapping
from pathlib import Path
from typing import override

from pydantic import JsonValue

from app.features.skills.entities import Skill, SkillMeta
from lib.observability import Logger

from .contracts import SkillSource
from .markdown import _meta, _split


class FilesystemSkillSource(SkillSource):
    def __init__(self, root: Path, glob: str, *, logger: Logger) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        if Path(glob).is_absolute() or '..' in Path(glob).parts:
            raise ValueError('skill glob must stay within its root')
        self.glob = glob
        self.logger = logger.bind(component='skills')

    def _path(self, path: str) -> Path:
        candidate = (self.root / path).resolve(strict=True)
        if not candidate.is_relative_to(self.root):
            raise ValueError('skill path escapes its root')
        return candidate

    @override
    async def list(self, scope: Mapping[str, JsonValue]) -> tuple[SkillMeta, ...]:
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        metas = []
        for path in sorted(self.root.glob(self.glob)):
            candidate = self._path(str(path))
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(self.root).as_posix()
            content = candidate.read_text(encoding='utf-8')
            try:
                metas.append(_meta(relative, content, logger=self.logger))
            except TypeError, ValueError:
                self.logger.warning('Skipping invalid skill frontmatter: %s', relative)
        return tuple(metas)

    @override
    async def load(self, path: str, scope: Mapping[str, JsonValue]) -> Skill:
        content = self._path(path).read_text(encoding='utf-8')
        return Skill(
            meta=_meta(path, content, logger=self.logger),
            body=_split(content)[1],
        )
