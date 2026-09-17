import json
from typing import Annotated, override

from pydantic import Field

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig, ToolRef
from app.features.agents.types import ToolOutput, ToolParams, TurnScope, TypedTool
from app.features.tools.config import ToolConfig, ToolConfigList
from app.features.tools.registry import ToolRegistry
from app.features.tools.sources import ConfiguredToolSource


class EchoParams(ToolParams):
    message: Annotated[str, Field()]
    user: Annotated[str, Field()]


class EchoTool(TypedTool[EchoParams]):
    name = 'echo'
    description = 'Echo a message'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return f'{params.user}:{params.message}'


def _source() -> ConfiguredToolSource:
    registry = ToolRegistry()
    registry.register(EchoTool)
    config = HuginnConfig(
        sections={'tools': ToolConfigList({'echo': ToolConfig(kind='echo')})},
    )
    return ConfiguredToolSource(config, registry)


async def test_configured_tool_source_binds_and_hides_turn_value() -> None:
    agent = AgentConfig(
        description='test agent',
        tools=(ToolRef(name='echo', bind={'user': 'handle'}),),
    )
    scope = TurnScope(
        actor='telegram:42',
        agent='scout',
        channel='telegram',
        chat='42',
        handle='alice',
    )

    harvest = await _source().collect(agent, scope)
    tool = harvest.tools[0][1]
    schema = tool.schema()

    assert isinstance(schema, dict)
    properties = schema['properties']
    assert isinstance(properties, dict)
    assert 'user' not in properties
    assert await tool.invoke(json.dumps({'message': 'hello', 'user': 'mallory'})) == (
        'alice:hello'
    )


async def test_configured_tool_source_omits_ref_with_missing_binding() -> None:
    agent = AgentConfig(
        description='test agent',
        tools=(ToolRef(name='echo', bind={'user': 'handle'}),),
    )
    scope = TurnScope(
        actor='telegram:42',
        agent='scout',
        channel='telegram',
        chat='42',
    )

    harvest = await _source().collect(agent, scope)

    assert harvest.tools == ()
