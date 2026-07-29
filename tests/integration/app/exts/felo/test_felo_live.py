from typing import Annotated

import pytest
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.exts.felo.clients import FeloClient
from app.features.agents.services import AgentRunner
from app.features.agents.types import ToolSet, TurnInput
from app.features.tools.websearch.repositories import WebSearchRepository
from app.features.tools.websearch.tool import WebSearchTool
from lib.http import HTTPClientManager
from lib.llm.adapters.openai import OpenAIAdapter
from lib.llm.client import ProviderConfig
from lib.temporal import ClockManager

pytestmark = pytest.mark.integration

_FELO_BASE_URL = 'https://openapi.felo.ai'
_OPENAI_MODEL = 'gpt-5-mini'


class LiveCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    felo_api_key: Annotated[SecretStr | None, Field()] = None
    openai_api_key: Annotated[SecretStr | None, Field()] = None


def require_felo_api_key() -> SecretStr:
    key = LiveCredentials().felo_api_key
    if key is None:
        pytest.skip('FELO_API_KEY not set')
    return key


def require_openai_api_key() -> SecretStr:
    key = LiveCredentials().openai_api_key
    if key is None:
        pytest.skip('OPENAI_API_KEY not set')
    return key


def build_client(http: HTTPClientManager, api_key: SecretStr) -> FeloClient:
    return FeloClient(http=http, api_key=api_key, base_url=_FELO_BASE_URL)


async def test_websearch_live() -> None:
    api_key = require_felo_api_key()

    result = await build_client(HTTPClientManager(), api_key).search(
        "today's weather in Taipei",
    )

    assert result.answer
    assert result.resources


async def test_agent_loop_with_websearch_live() -> None:
    felo_api_key = require_felo_api_key()
    openai_api_key = require_openai_api_key()

    http = HTTPClientManager()
    tools = ToolSet()
    repository = WebSearchRepository(http)
    tools.add('websearch', WebSearchTool(repository, felo_api_key, _FELO_BASE_URL))
    llm = OpenAIAdapter(
        ProviderConfig(
            name='openai',
            adapter='openai',
            api_key=openai_api_key,
            models=(_OPENAI_MODEL,),
        ),
        http,
    )
    turn = TurnInput(
        instructions=(
            'Use the websearch tool to find current information before answering.'
        ),
        prompts=("What is today's weather in Taipei? Answer briefly.",),
        model=_OPENAI_MODEL,
    )

    fragments = []

    async def collect(fragment: str) -> None:
        fragments.append(fragment)

    output = await AgentRunner(llm, tools, ClockManager()).run(turn, collect)

    executions = [record for step in output.steps for record in step.tools]

    assert output.completed
    assert output.text == ''.join(fragments)
    assert any(record.name == 'websearch' and record.ok for record in executions)
