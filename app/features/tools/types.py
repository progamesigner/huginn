from abc import ABC, abstractmethod
from typing import override

from pydantic import BaseModel, ConfigDict, JsonValue

type ToolOutput = str


class ToolParams(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    async def invoke(self, arguments: str) -> ToolOutput: ...

    @abstractmethod
    def schema(self) -> JsonValue: ...


class TypedTool[P: ToolParams](Tool):
    params: type[P]

    @abstractmethod
    async def run(self, params: P) -> ToolOutput: ...

    @override
    async def invoke(self, arguments: str) -> ToolOutput:
        return await self.run(self.params.model_validate_json(arguments))

    @override
    def schema(self) -> JsonValue:
        return self.params.model_json_schema()
