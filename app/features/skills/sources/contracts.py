from abc import ABC, abstractmethod

from app.features.skills.types import Skill, SkillMeta


class SkillSource(ABC):
    @abstractmethod
    async def list(self) -> tuple[SkillMeta, ...]: ...

    @abstractmethod
    async def load(self, path: str) -> Skill: ...
