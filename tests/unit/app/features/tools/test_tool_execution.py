from typing import Annotated, override

from pydantic import Field

from app.contracts.channels import ReplyArtifact
from app.features.agents.contracts import ToolOutput, ToolParams, ToolResult, TypedTool
from app.features.agents.tooling import ToolSet
from lib.llm import ToolCallSegment
from tests.helpers.logging import make_logger


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
    tools = ToolSet(logger=make_logger())
    tools.add('echo', EchoTool())
    tools.add('boom', BoomTool())
    return tools


async def test_execute_returns_tool_output_for_valid_arguments() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='echo', arguments='{"text": "hi"}')

    result, _ = await tools.execute(call)

    assert result.tool_call_id == 'call_1'
    assert result.content == 'hi'
    assert result.error is False


async def test_execute_reports_invalid_json_as_error_result() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='echo', arguments='{not json')

    result, _ = await tools.execute(call)

    assert result.error is True
    assert 'json' in result.content.lower()


async def test_execute_reports_missing_field_as_error_result() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='echo', arguments='{}')

    result, _ = await tools.execute(call)

    assert result.error is True
    assert 'text' in result.content


# NOTE: the reason has to travel -- 'failed' alone leaves the model unable to tell an
# unreachable server from a call it built wrong, so it can neither retry nor route around
async def test_execute_reports_why_a_tool_raised_not_only_that_it_did() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='boom', arguments='{"text": "x"}')

    result, _ = await tools.execute(call)

    assert result.error is True
    assert "tool 'boom' failed" in result.content
    assert 'muninn: not connected' in result.content


class SilentTimeoutTool(TypedTool[EchoParams]):
    name = 'silent'
    description = 'Raises an error that carries no message'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        raise TimeoutError


# NOTE: an httpx2 timeout stringifies to '', which used to leave the model reading
# "failed: " with nothing after it -- the type is then the only reason left
async def test_execute_names_the_error_type_when_the_message_is_empty() -> None:
    tools = ToolSet(logger=make_logger())
    tools.add('silent', SilentTimeoutTool())
    call = ToolCallSegment(id='call_1', name='silent', arguments='{"text": "x"}')

    result, _ = await tools.execute(call)

    assert result.error is True
    assert result.content == "tool 'silent' failed: TimeoutError"


async def test_execute_reports_unknown_tool_with_available_list() -> None:
    tools = make_toolset()
    call = ToolCallSegment(id='call_1', name='missing', arguments='{}')

    result, _ = await tools.execute(call)

    assert result.error is True
    assert 'boom, echo' in result.content


_ARTIFACT = ReplyArtifact(
    blob='0190a1b2-0000-7000-8000-000000000001',
    kind='document',
    mime='text/csv',
    size=3,
    data=b'a,b',
    name='table.csv',
)


class FileTool(TypedTool[EchoParams]):
    name = 'file'
    description = 'Produce a file'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return ToolResult(text=params.text, artifacts=(_ARTIFACT,))


async def test_execute_keeps_a_plain_answer_free_of_artifacts() -> None:
    call = ToolCallSegment(id='call_1', name='echo', arguments='{"text": "hi"}')

    result, artifacts = await make_toolset().execute(call)

    assert result.content == 'hi'
    assert artifacts == ()


async def test_execute_splits_an_artifact_from_the_text_the_model_reads() -> None:
    tools = ToolSet(logger=make_logger())
    tools.add('file', FileTool())
    call = ToolCallSegment(id='call_1', name='file', arguments='{"text": "made"}')

    result, artifacts = await tools.execute(call)

    assert (result.content, result.error) == ('made', False)
    assert artifacts == (_ARTIFACT,)
