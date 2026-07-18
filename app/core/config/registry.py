from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from pydantic import BaseModel

from .exceptions import DuplicateSectionError, UnknownSectionError

type SectionDescriber = Callable[[HuginnConfig], list[str]]
type SectionValidator = Callable[[HuginnConfig], None]
type SectionModel = type[BaseModel] | Callable[[], type[BaseModel]]
type Sections = dict[str, BaseModel]
type Specs = dict[str, SectionSpec]


class SectionRegistry:
    specs: Specs

    def __init__(self) -> None:
        self.specs = cast(Specs, {})

    def register(self, spec: SectionSpec) -> None:
        if spec.key in self.specs:
            raise DuplicateSectionError(
                f'config section {spec.key!r} is already registered'
            )

        self.specs[spec.key] = spec

    def schema(self) -> dict[str, object]:
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

    def __iter__(self) -> Iterator[SectionSpec]:
        return iter(self.specs.values())


@dataclass(frozen=True)
class SectionSpec:
    key: str
    model: SectionModel
    required: bool = False
    describe: SectionDescriber | None = None
    validate: SectionValidator | None = None

    @cached_property
    def resolved(self) -> type[BaseModel]:
        # NOTE: thunk models resolve lazily, after every provider has contributed
        model = self.model
        resolved = model if isinstance(model, type) else model()
        return cast(type[BaseModel], resolved)


@dataclass(frozen=True)
class HuginnConfig:
    sections: Sections

    def section[S: BaseModel](self, key: str, model: type[S]) -> S:
        if key not in self.sections:
            known = ', '.join(sorted(self.sections)) or 'none'
            raise UnknownSectionError(f'unknown config section {key!r}; known: {known}')

        value = self.sections[key]
        if not isinstance(value, model):
            raise UnknownSectionError(
                f'config section {key!r} is {type(value).__name__},'
                f' not {model.__name__}'
            )

        return value
