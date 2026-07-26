from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from pydantic import BaseModel, JsonValue

from .exceptions import DuplicateSectionError, UnknownSectionError

type SectionDescriber = Callable[[HuginnConfig], list[str]]
type SectionValidator = Callable[[HuginnConfig], None]
type ModelThunk = type[BaseModel] | Callable[[], type[BaseModel]]
type Sections = dict[str, BaseModel]
type Specs = dict[str, SectionSpec[BaseModel]]


class SectionRegistry:
    specs: Specs

    def __init__(self) -> None:
        self.specs = {}

    def register(self, spec: SectionSpec[BaseModel]) -> None:
        if spec.key in self.specs:
            raise DuplicateSectionError(
                f'config section {spec.key!r} is already registered',
            )

        self.specs[spec.key] = spec

    def schema(self) -> dict[str, JsonValue]:
        return {
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                spec.key: spec.resolved.model_json_schema() for spec in self
            },
            'required': [spec.key for spec in self if spec.required],
        }

    def __contains__(self, key: str) -> bool:
        return key in self.specs

    def __iter__(self) -> Iterator[SectionSpec[BaseModel]]:
        return iter(self.specs.values())


@dataclass(frozen=True)
class SectionSpec[S: BaseModel]:
    key: str
    model: type[S]
    thunk: ModelThunk | None = None
    required: bool = False
    describe: SectionDescriber | None = None
    validate: SectionValidator | None = None

    @cached_property
    def resolved(self) -> type[BaseModel]:
        # NOTE: tools set a thunk that builds its union from the registry
        thunk = self.thunk or self.model
        resolved = thunk if isinstance(thunk, type) else thunk()
        return cast(type[BaseModel], resolved)


@dataclass(frozen=True)
class HuginnConfig:
    sections: Sections

    def section[S: BaseModel](self, spec: SectionSpec[S]) -> S:
        if spec.key not in self.sections:
            known = ', '.join(sorted(self.sections)) or 'none'
            raise UnknownSectionError(
                f'unknown config section {spec.key!r}; known: {known}',
            )

        value = self.sections[spec.key]
        if not isinstance(value, spec.model):
            raise UnknownSectionError(
                f'config section {spec.key!r} is {type(value).__name__},'
                f' not {spec.model.__name__}',
            )

        return value
