from textwrap import shorten
from typing import Annotated, Literal, override

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from app.exts.felo.client import FeloClient, SearchResult
from lib.http import HTTPClientContract

from .config import ToolConfig
from .exceptions import ToolError
from .types import Tool, ToolOutput, ToolParams


class WebSearchToolConfig(ToolConfig):
    kind: Annotated[Literal['websearch'], Field()]
    api_key: Annotated[SecretStr, Field()]
    base_url: Annotated[HttpUrl, Field()] = HttpUrl('https://openapi.felo.ai')


_SNIPPET_WIDTH = 200


class WebSearchParams(ToolParams):
    query: Annotated[
        str,
        Field(description='What to search the web for, phrased as a question'),
    ]
    max_results: Annotated[
        int,
        Field(ge=1, le=10, description='Maximum number of sources to include'),
    ] = 5


def _format_result(result: SearchResult, max_results: int) -> str:
    lines = [result.answer.strip()] if result.answer.strip() else []

    if resources := result.resources[:max_results]:
        lines.extend(('', 'Sources:') if lines else ('Sources:',))
        for index, resource in enumerate(resources, start=1):
            lines.append(f'{index}. {resource.title}')
            lines.append(f'   {resource.link}')
            if snippet := resource.snippet.strip():
                lines.append(f'   {shorten(snippet, width=_SNIPPET_WIDTH)}')

    return '\n'.join(lines) if lines else 'No results found.'


class WebSearchTool(Tool[WebSearchParams]):
    name = 'websearch'
    description = (
        'Search the web for current information. Returns a synthesized answer'
        ' followed by the sources it was drawn from.'
    )
    params = WebSearchParams

    client: FeloClient

    def __init__(self, client: FeloClient) -> None:
        self.client = client

    @override
    async def run(self, params: WebSearchParams) -> ToolOutput:
        result = await self.client.search(params.query)
        return _format_result(result, params.max_results)


def build_websearch(http: HTTPClientContract, config: BaseModel) -> WebSearchTool:
    if not isinstance(config, WebSearchToolConfig):
        raise ToolError(
            f'websearch requires WebSearchToolConfig, got {type(config).__name__}'
        )

    return WebSearchTool(
        FeloClient(
            http=http,
            api_key=config.api_key,
            base_url=str(config.base_url),
        )
    )
