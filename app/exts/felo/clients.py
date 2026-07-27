from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from lib.http import HTTPClientContract


class FeloValue(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)


class SearchResource(FeloValue):
    link: Annotated[str, Field()]
    title: Annotated[str, Field()]
    snippet: Annotated[str, Field()] = ''


class SearchResult(FeloValue):
    answer: Annotated[str, Field()] = ''
    resources: Annotated[tuple[SearchResource, ...], Field()] = ()


class SearchResponse(FeloValue):
    data: Annotated[SearchResult, Field()]


class FeloClient:
    http: HTTPClientContract
    base_url: str
    headers: dict[str, str]

    def __init__(
        self,
        http: HTTPClientContract,
        api_key: SecretStr,
        base_url: str,
    ) -> None:
        self.http = http
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key.get_secret_value()}'}

    async def search(self, query: str) -> SearchResult:
        async with self.http.connect(self.base_url, self.headers) as client:
            response = await client.post('/v2/chat', json={'query': query})
            return SearchResponse.model_validate_json(
                response.raise_for_status().content,
            ).data
