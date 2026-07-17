from typing import Annotated, override

from pydantic import Field

from app.core.tools import Tool, ToolOutput, ToolParams, ToolRegistry
from lib.llm import ToolCallSegment


class EchoParams(ToolParams):
    text: Annotated[str, Field()]


class EchoTool(Tool[EchoParams]):
    name = 'echo'
    description = 'Echo the given text back'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return params.text


class BoomTool(Tool[EchoParams]):
    name = 'boom'
    description = 'Always raises'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        raise RuntimeError('boom')


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool)
    registry.register(BoomTool)
    return registry


async def test_execute_returns_tool_output_for_valid_arguments() -> None:
    registry = make_registry()
    call = ToolCallSegment(id='call_1', name='echo', arguments='{"text": "hi"}')

    result = await registry.execute(call)

    assert result.tool_call_id == 'call_1'
    assert result.content == 'hi'
    assert result.error is False


async def test_execute_reports_malformed_json_as_tool_error() -> None:
    registry = make_registry()
    call = ToolCallSegment(id='call_2', name='echo', arguments='{not json')

    result = await registry.execute(call)

    assert result.tool_call_id == 'call_2'
    assert result.error is True


async def test_execute_reports_missing_required_field_as_tool_error() -> None:
    registry = make_registry()
    call = ToolCallSegment(id='call_3', name='echo', arguments='{}')

    result = await registry.execute(call)

    assert result.tool_call_id == 'call_3'
    assert result.error is True


async def test_execute_reports_tool_internal_exception_as_tool_error() -> None:
    registry = make_registry()
    call = ToolCallSegment(id='call_4', name='boom', arguments='{"text": "hi"}')

    result = await registry.execute(call)

    assert result.tool_call_id == 'call_4'
    assert result.error is True


async def test_execute_reports_unknown_tool_name_as_tool_error() -> None:
    registry = make_registry()
    call = ToolCallSegment(id='call_5', name='missing', arguments='{}')

    result = await registry.execute(call)

    assert result.tool_call_id == 'call_5'
    assert result.error is True


async def test_execute_reports_unexpected_argument_as_tool_error() -> None:
    registry = make_registry()
    call = ToolCallSegment(
        id='call_6',
        name='echo',
        arguments='{"text": "hi", "shout": true}',
    )

    result = await registry.execute(call)

    assert result.tool_call_id == 'call_6'
    assert result.error is True
    assert 'shout' in result.content
