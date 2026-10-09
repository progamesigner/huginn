from typing import Annotated, override
from uuid import UUID

from pydantic import BaseModel, Field

from app.features.agents.contracts import ToolParams, TypedTool
from app.features.agents.entities import TurnScope
from app.features.blobs.contracts import BlobServiceContract
from app.features.tools.exceptions import ToolError

from .config import DocumentToolConfig


class ReadDocumentParams(ToolParams):
    blob: Annotated[UUID, Field(description='The document blob identifier')]
    offset: Annotated[int, Field(ge=0, description='Character offset')] = 0
    limit: Annotated[
        int,
        Field(gt=0, le=20_000, description='Characters to return'),
    ] = 4_000


class ReadDocumentTool(TypedTool[ReadDocumentParams]):
    name = 'read_document'
    description = (
        'Read a bounded section of an extracted document from this conversation.'
        ' The first line gives the character range, the total length and the offset'
        ' to continue from.'
    )
    params = ReadDocumentParams

    def __init__(
        self,
        blobs: BlobServiceContract,
        allowed: frozenset[str],
        max_chars: int,
    ) -> None:
        self.blobs = blobs
        self.allowed = allowed
        self.max_chars = max_chars

    @override
    async def run(self, params: ReadDocumentParams) -> str:
        if str(params.blob) not in self.allowed:
            raise ToolError('document blob is not available in this conversation')

        blob = await self.blobs.load(params.blob)
        if blob is None or blob.data is None:
            raise ToolError('document blob was not found')
        if blob.mime not in {'text/plain', 'text/markdown'}:
            raise ToolError('blob is not an extracted text document')

        text = blob.data.decode('utf-8', errors='replace')
        start = min(params.offset, len(text))
        end = min(start + min(params.limit, self.max_chars), len(text))
        position = f'next offset {end}' if end < len(text) else 'end of document'
        return (
            f'[characters {start}-{end} of {len(text)}; {position}]\n{text[start:end]}'
        )


def build_read_document(
    blobs: BlobServiceContract,
    config: BaseModel,
    scope: TurnScope,
) -> ReadDocumentTool:
    if not isinstance(config, DocumentToolConfig):
        raise ToolError(
            f'read_document requires DocumentToolConfig, got {type(config).__name__}',
        )

    return ReadDocumentTool(blobs, scope.document_blobs, config.max_chars)
