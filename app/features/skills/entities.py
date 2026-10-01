from typing import Annotated

from pydantic import Field

from app.core.schema import AppValue


class SkillMeta(AppValue):
    name: Annotated[str, Field()]
    description: Annotated[str, Field()]
    path: Annotated[str, Field()]


class Skill(AppValue):
    meta: Annotated[SkillMeta, Field()]
    body: Annotated[str, Field()]
