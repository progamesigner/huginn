from datetime import timedelta
from typing import override
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.agents.entities import TurnScope
from app.features.agents.tooling import ToolSet
from app.features.blobs.contracts import BlobServiceContract
from app.features.blobs.entities import Blob, BlobKind
from app.features.tools.document.config import DocumentToolConfig
from app.features.tools.document.tool import ReadDocumentTool, build_read_document
from app.features.tools.exceptions import ToolError
from app.features.tools.websearch.config import WebSearchToolConfig
from lib.llm import ToolCallSegment
from tests.helpers.logging import make_logger

_SCOPE = TurnScope(actor='user:1', agent='scout', channel='support_bot', chat='1')


class Blobs(BlobServiceContract):
    def __init__(self, blob: Blob | None) -> None:
        self.blob = blob

    @override
    async def save(
        self,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob:
        raise NotImplementedError

    @override
    async def grant(self, id: UUID, owner: str) -> None:
        raise NotImplementedError

    @override
    async def granted(self, id: UUID, owner: str) -> bool:
        raise NotImplementedError

    @override
    async def load(self, id: UUID) -> Blob | None:
        return self.blob if self.blob and self.blob.id == id else None

    @override
    async def store(
        self,
        transaction: AsyncSession,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob:
        raise NotImplementedError

    @override
    async def head(self, id: UUID) -> Blob | None:
        return None


def _text(id_: UUID, data: bytes, mime: str = 'text/markdown') -> Blob:
    return Blob(
        id=id_,
        sha256='hash',
        mime=mime,
        size=len(data),
        kind='document',
        data=data,
    )


def _tool(
    blob: Blob | None,
    *allowed: UUID,
    max_chars: int = 8_000,
) -> ReadDocumentTool:
    return ReadDocumentTool(
        Blobs(blob),
        frozenset(str(id_) for id_ in allowed),
        max_chars,
    )


async def test_reads_an_allowed_text_blob_and_says_it_reached_the_end() -> None:
    blob_id = uuid4()
    tool = _tool(_text(blob_id, b'hello'), blob_id)

    assert await tool.invoke(f'{{"blob":"{blob_id}"}}') == (
        '[characters 0-5 of 5; end of document]\nhello'
    )


async def test_rejects_blob_outside_conversation() -> None:
    blob_id = uuid4()
    tool = _tool(_text(blob_id, b'hello'))

    with pytest.raises(ToolError, match='not available'):
        await tool.invoke(f'{{"blob":"{blob_id}"}}')


async def test_rejects_missing_blob() -> None:
    blob_id = uuid4()
    tool = _tool(None, blob_id)

    with pytest.raises(ToolError, match='not found'):
        await tool.invoke(f'{{"blob":"{blob_id}"}}')


async def test_rejects_non_text_blob() -> None:
    blob_id = uuid4()
    tool = _tool(_text(blob_id, b'hello', mime='application/pdf'), blob_id)

    with pytest.raises(ToolError, match='extracted text'):
        await tool.invoke(f'{{"blob":"{blob_id}"}}')


async def test_applies_offset_and_configured_limit_and_names_the_next_offset() -> None:
    blob_id = uuid4()
    tool = _tool(_text(blob_id, b'0123456789', mime='text/plain'), blob_id, max_chars=3)

    assert await tool.invoke(f'{{"blob":"{blob_id}","offset":2,"limit":9}}') == (
        '[characters 2-5 of 10; next offset 5]\n234'
    )


async def test_an_offset_past_the_end_reads_nothing() -> None:
    blob_id = uuid4()
    tool = _tool(_text(blob_id, b'abc'), blob_id)

    assert await tool.invoke(f'{{"blob":"{blob_id}","offset":7}}') == (
        '[characters 3-3 of 3; end of document]\n'
    )


def test_the_builder_scopes_the_tool_to_the_turns_documents() -> None:
    blob_id = uuid4()
    scope = TurnScope(
        actor='user:1',
        agent='scout',
        channel='support_bot',
        chat='1',
        document_blobs=frozenset({str(blob_id)}),
    )

    tool = build_read_document(
        Blobs(None),
        DocumentToolConfig(kind='read_document', max_chars=100),
        scope,
    )

    assert tool.allowed == frozenset({str(blob_id)})
    assert tool.max_chars == 100


def test_the_builder_refuses_a_foreign_config() -> None:
    config = WebSearchToolConfig.model_validate({'kind': 'websearch', 'api_key': 'k'})

    with pytest.raises(ToolError, match='DocumentToolConfig'):
        build_read_document(Blobs(None), config, _SCOPE)


async def test_tool_set_returns_a_safe_error_without_aborting() -> None:
    blob_id = uuid4()
    result, _ = await ToolSet(
        {'read_document': _tool(None, blob_id)},
        logger=make_logger(),
    ).execute(
        ToolCallSegment(
            id='call-1',
            name='read_document',
            arguments=f'{{"blob":"{blob_id}"}}',
        ),
    )

    assert result.error is True
    assert result.content == "tool 'read_document' failed: document blob was not found"
