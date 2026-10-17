from collections.abc import Mapping
from typing import override

from pydantic import JsonValue

from app.features.assets.contracts import AssetsServiceContract
from app.features.skills.entities import Skill, SkillMeta

from .contracts import SkillSource


def _meta(name: str, path: str, front: Mapping[str, JsonValue]) -> SkillMeta:
    return SkillMeta(
        name=name,
        path=path,
        description=str(front.get('description') or name),
    )


class AssetSkillSource(SkillSource):
    cache_inventory = False

    def __init__(self, assets: AssetsServiceContract, name: str) -> None:
        self.assets = assets
        self.name = name

    @override
    async def list(self, scope: Mapping[str, JsonValue]) -> tuple[SkillMeta, ...]:
        return tuple(
            _meta(item.name, item.path, item.frontmatter)
            for item in await self.assets.inventory(self.name, 'skill', scope)
        )

    @override
    async def load(self, path: str, scope: Mapping[str, JsonValue]) -> Skill:
        meta = next(
            (
                item
                for item in await self.assets.inventory(self.name, 'skill', scope)
                if item.path == path
            ),
            None,
        )
        if meta is None:
            raise FileNotFoundError(path)
        asset = await self.assets.load(self.name, 'skill', meta.name, scope)
        if asset is None:
            raise FileNotFoundError(path)
        return Skill(
            meta=_meta(asset.meta.name, path, asset.meta.frontmatter),
            body=asset.body,
        )
