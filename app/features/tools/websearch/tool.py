from textwrap import shorten
from typing import Annotated, override

from pydantic import BaseModel, Field, SecretStr

from app.features.tools.exceptions import ToolError
from app.features.tools.types import ToolOutput, ToolParams, TypedTool

from .config import WebSearchToolConfig
from .repositories import WebSearchRepository, WebSearchResult

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


def _format_result(result: WebSearchResult, max_results: int) -> str:
    lines = [result.answer.strip()] if result.answer.strip() else []

    if sources := result.sources[:max_results]:
        lines.extend(('', 'Sources:') if lines else ('Sources:',))
        for index, source in enumerate(sources, start=1):
            lines.append(f'{index}. {source.title}')
            lines.append(f'   {source.link}')
            if snippet := source.snippet.strip():
                lines.append(f'   {shorten(snippet, width=_SNIPPET_WIDTH)}')

    return '\n'.join(lines) if lines else 'No results found.'


class WebSearchTool(TypedTool[WebSearchParams]):
    name = 'websearch'
    description = (
        'Search the web for current information. Returns a synthesized answer'
        ' followed by the sources it was drawn from.'
    )
    params = WebSearchParams

    repository: WebSearchRepository
    api_key: SecretStr
    base_url: str

    def __init__(
        self,
        repository: WebSearchRepository,
        api_key: SecretStr,
        base_url: str,
    ) -> None:
        self.repository = repository
        self.api_key = api_key
        self.base_url = base_url

    @override
    async def run(self, params: WebSearchParams) -> ToolOutput:
        result = await self.repository.search(params.query, self.api_key, self.base_url)
        return _format_result(result, params.max_results)


def build_websearch(
    repository: WebSearchRepository,
    config: BaseModel,
) -> WebSearchTool:
    if not isinstance(config, WebSearchToolConfig):
        raise ToolError(
            f'websearch requires WebSearchToolConfig, got {type(config).__name__}',
        )

    return WebSearchTool(repository, config.api_key, str(config.base_url))
