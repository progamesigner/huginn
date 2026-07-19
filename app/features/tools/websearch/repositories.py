from dataclasses import dataclass

from pydantic import SecretStr

from app.exts.felo.client import FeloClient
from lib.http import HTTPClientContract


@dataclass(frozen=True)
class WebSearchSource:
    title: str
    link: str
    snippet: str = ''


@dataclass(frozen=True)
class WebSearchResult:
    answer: str = ''
    sources: tuple[WebSearchSource, ...] = ()


class WebSearchRepository:
    http: HTTPClientContract

    def __init__(self, http: HTTPClientContract) -> None:
        self.http = http

    async def search(
        self,
        query: str,
        api_key: SecretStr,
        base_url: str,
    ) -> WebSearchResult:
        client = FeloClient(http=self.http, api_key=api_key, base_url=base_url)
        result = await client.search(query)
        return WebSearchResult(
            answer=result.answer,
            sources=tuple(
                WebSearchSource(
                    title=resource.title,
                    link=resource.link,
                    snippet=resource.snippet,
                )
                for resource in result.resources
            ),
        )
