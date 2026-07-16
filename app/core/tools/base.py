from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

type ToolOutput = str


class ToolParams(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Tool[P: ToolParams](ABC):
    name: str
    description: str
    params: type[P]

    @abstractmethod
    async def run(self, params: P) -> ToolOutput: ...
