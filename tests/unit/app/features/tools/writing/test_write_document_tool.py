import json
from datetime import timedelta

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.exts.telegram.clients import TelegramClient
from app.exts.telegram.gateways import TelegramReply
from app.features.agents.tooling import ToolSet
from app.features.tools.artifacts import ArtifactWriter
from app.features.tools.exceptions import ToolError
from app.features.tools.websearch.config import WebSearchToolConfig
from app.features.tools.writing.config import WritingToolConfig
from app.features.tools.writing.tool import (
    WriteDocumentParams,
    WriteDocumentTool,
    build_write_document,
)
from lib.http import HTTPClientManager
from lib.llm import ToolCallSegment
from tests.helpers.blobs import MemoryBlobs
from tests.helpers.clock import FakeClock
from tests.helpers.logging import make_logger

_TELEGRAM = 'https://api.telegram.org/bottelegram-token'
_TTL = timedelta(days=30)


def make_tool(blobs: MemoryBlobs, max_bytes: int | None = None) -> WriteDocumentTool:
    config = WritingToolConfig.model_validate(
        {
            'kind': 'write_document',
            **({} if max_bytes is None else {'max_bytes': max_bytes}),
        },
    )
    return build_write_document(ArtifactWriter(blobs, _TTL), config)


def call(arguments: dict[str, object]) -> ToolCallSegment:
    return ToolCallSegment(id='c', name='writer', arguments=json.dumps(arguments))


async def test_a_written_file_is_stored_and_handed_over() -> None:
    blobs = MemoryBlobs()

    result = await make_tool(blobs).run(
        WriteDocumentParams(format='md', filename='notes', content='# hi'),
    )

    assert result.text == 'Wrote notes.md and sent it to the user.'
    (artifact,) = result.artifacts
    assert (artifact.kind, artifact.mime, artifact.name, artifact.data) == (
        'document',
        'text/markdown',
        'notes.md',
        b'# hi',
    )
    assert artifact.blob in {str(blob) for blob in blobs.blobs}


async def test_numbers_from_the_model_stay_numbers_in_the_sheet() -> None:
    params = WriteDocumentParams.model_validate_json(
        json.dumps(
            {
                'format': 'xlsx',
                'filename': 'sales',
                'sheets': [{'name': 'Q1', 'rows': [['Jan', 1200, 3.5, '42', None]]}],
            },
        ),
    )

    assert params.sheets[0].rows == (('Jan', 1200, 3.5, '42', None),)


async def test_a_file_over_the_limit_goes_back_to_the_model_as_an_error() -> None:
    tools = ToolSet(
        {'writer': make_tool(MemoryBlobs(), max_bytes=4)},
        logger=make_logger(),
    )

    result, artifacts = await tools.execute(
        call({'format': 'md', 'filename': 'big', 'content': 'too long for four'}),
    )

    assert result.error is True
    assert 'over the 4 byte limit' in result.content
    assert artifacts == ()


async def test_a_format_missing_its_input_goes_back_to_the_model() -> None:
    tools = ToolSet({'writer': make_tool(MemoryBlobs())}, logger=make_logger())

    result, _ = await tools.execute(call({'format': 'xlsx', 'filename': 'x'}))

    assert result.error is True
    assert 'at least one sheet' in result.content


def test_build_write_document_refuses_another_tools_config() -> None:
    other = WebSearchToolConfig.model_validate({'kind': 'websearch', 'api_key': 'k'})

    with pytest.raises(ToolError, match='WritingToolConfig'):
        build_write_document(ArtifactWriter(MemoryBlobs(), _TTL), other)


@respx.mock
async def test_a_written_file_reaches_telegram_as_a_document_with_a_safe_name() -> None:
    document = respx.post(f'{_TELEGRAM}/sendDocument').mock(
        return_value=httpx.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'message_id': 2,
                    'date': 1_721_600_000,
                    'chat': {'id': 5, 'type': 'private'},
                },
            },
        ),
    )
    result = await make_tool(MemoryBlobs()).run(
        WriteDocumentParams(
            format='csv',
            filename='../../etc/passwd\nX',
            content='a,b',
        ),
    )
    reply = TelegramReply(
        TelegramClient(http=HTTPClientManager(), bot_token=SecretStr('telegram-token')),
        5,
        FakeClock(),
        logger=make_logger(),
    )

    await reply.on_artifact(result.artifacts[0])

    assert b'filename="passwd_X.csv"' in document.calls.last.request.content
