import asyncio
from typing import Annotated, override

from pydantic import BaseModel, Field

from app.features.agents.contracts import ToolParams, ToolResult, TypedTool
from app.features.tools.artifacts import ArtifactWriter
from app.features.tools.exceptions import ToolError

from .config import WritingToolConfig
from .renderers import Cell, DocumentFormat, Rendered, RenderError, Sheet, render


class SheetParams(ToolParams):
    name: Annotated[str, Field(description='The sheet tab name')]
    rows: Annotated[
        tuple[tuple[Cell, ...], ...],
        Field(
            description=(
                'Rows of cells, the first usually a header. Give numbers as numbers,'
                ' not strings, so the spreadsheet can add them up.'
            ),
        ),
    ]


class WriteDocumentParams(ToolParams):
    format: Annotated[
        DocumentFormat,
        Field(description='md, csv, docx or xlsx'),
    ]
    filename: Annotated[str, Field(description='A short file name for the user')]
    content: Annotated[
        str,
        Field(
            description=(
                'The text for md, csv or docx; docx reads Markdown headings, lists,'
                ' tables, bold, italic and code'
            ),
        ),
    ] = ''
    sheets: Annotated[
        tuple[SheetParams, ...],
        Field(description='Required for xlsx; for csv, the first sheet becomes rows'),
    ] = ()


class WriteDocumentTool(TypedTool[WriteDocumentParams]):
    name = 'write_document'
    description = (
        'Write a file for the user to download: Markdown notes, a CSV, a Word'
        ' document or an Excel workbook. The file is sent to them directly.'
    )
    params = WriteDocumentParams

    writer: ArtifactWriter
    config: WritingToolConfig

    def __init__(self, writer: ArtifactWriter, config: WritingToolConfig) -> None:
        self.writer = writer
        self.config = config

    @override
    async def run(self, params: WriteDocumentParams) -> ToolResult:
        rendered = await self._render(params)
        if len(rendered.data) > self.config.max_bytes:
            raise ToolError(
                f'{rendered.name} came to {len(rendered.data)} bytes, over the'
                f' {self.config.max_bytes} byte limit; write less into it',
            )

        artifact = await self.writer.write(
            rendered.mime,
            rendered.data,
            kind='document',
            name=rendered.name,
        )
        return ToolResult(
            text=f'Wrote {rendered.name} and sent it to the user.',
            artifacts=(artifact,),
        )

    async def _render(self, params: WriteDocumentParams) -> Rendered:
        sheets = tuple(
            Sheet(name=sheet.name, rows=sheet.rows) for sheet in params.sheets
        )
        try:
            return await asyncio.to_thread(
                render,
                params.format,
                params.filename,
                params.content,
                sheets,
            )
        except RenderError as error:
            raise ToolError(str(error)) from error


def build_write_document(
    writer: ArtifactWriter,
    config: BaseModel,
) -> WriteDocumentTool:
    if not isinstance(config, WritingToolConfig):
        raise ToolError(
            f'write_document requires WritingToolConfig, got {type(config).__name__}',
        )

    return WriteDocumentTool(writer, config)
