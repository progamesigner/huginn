from dataclasses import dataclass


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    path: str


@dataclass(frozen=True)
class Skill:
    meta: SkillMeta
    body: str
