from typing import Annotated, override

from pydantic import Field

from app.features.agents.contracts import ToolOutput, ToolParams, TypedTool
from app.features.agents.tooling import ToolSet
from lib.llm import ToolCallSegment


class EchoParams(ToolParams):
    text: Annotated[str, Field()]


class EchoTool(TypedTool[EchoParams]):
    name = 'echo'
    description = 'Echo the given text back'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return params.text


class BoomTool(TypedTool[EchoParams]):
    name = 'boom'
    description = 'Always raises'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        raise RuntimeError('muninn: not connected')


def make_toolset() -> ToolSet:
    tools = ToolSet()
    tools.add('echo', EchoTool())
    tools.add('boom', BoomTool())
    return tools


async def test_execute_returns_tool_output_for_valid_arguments() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='echo', arguments='{"text": "hi"}')

    result = await tools.execute(call)

    assert result.tool_call_id == 'call_1'
    assert result.content == 'hi'
    assert result.error is False


async def test_execute_reports_invalid_json_as_error_result() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='echo', arguments='{not json')

    result = await tools.execute(call)

    assert result.error is True
    assert 'json' in result.content.lower()


async def test_execute_reports_missing_field_as_error_result() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='echo', arguments='{}')

    result = await tools.execute(call)

    assert result.error is True
    assert 'text' in result.content


# NOTE: the reason has to travel -- 'failed' alone leaves the model unable to tell an
# unreachable server from a call it built wrong, so it can neither retry nor route around
async def test_execute_reports_why_a_tool_raised_not_only_that_it_did() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='boom', arguments='{"text": "x"}')

    result = await tools.execute(call)

    assert result.error is True
    assert "tool 'boom' failed" in result.content
    assert 'muninn: not connected' in result.content


async def test_execute_reports_unknown_tool_with_available_list() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='missing', arguments='{}')

    result = await tools.execute(call)

    assert result.error is True
    assert 'boom, echo' in result.content
