from abc import ABC, abstractmethod
from collections.abc import Mapping

from pydantic import JsonValue

from app.features.skills.types import Skill, SkillMeta


class SkillSource(ABC):
    @abstractmethod
    async def list(self, scope: Mapping[str, JsonValue]) -> tuple[SkillMeta, ...]: ...

    @abstractmethod
    async def load(self, path: str, scope: Mapping[str, JsonValue]) -> Skill: ...
