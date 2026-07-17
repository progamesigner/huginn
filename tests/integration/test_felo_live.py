from typing import Annotated

import pytest
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.exts.felo.client import FeloClient
from app.exts.felo.settings import FeloExtSettings
from app.features.agents.services import AgentRunner, TurnInput
from app.features.tools import ToolRegistry, WebSearchTool
from lib.http import HTTPClientManager
from lib.llm.adapters.openai import OpenAIAdapter
from lib.llm.client import ProviderConfig

pytestmark = pytest.mark.integration

_OPENAI_MODEL = 'gpt-5-mini'


class LiveCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    openai_api_key: Annotated[SecretStr | None, Field()] = None


def require_felo_settings() -> FeloExtSettings:
    try:
        return FeloExtSettings()
    except ValidationError:
        pytest.skip('EXT_FELO_API_KEY not set')


def require_openai_api_key() -> SecretStr:
    key = LiveCredentials().openai_api_key
    if key is None:
        pytest.skip('OPENAI_API_KEY not set')
    return key


def build_client(http: HTTPClientManager, settings: FeloExtSettings) -> FeloClient:
    return FeloClient(
        http=http,
        api_key=settings.api_key,
        base_url=str(settings.base_url),
    )


async def test_websearch_live() -> None:
    settings = require_felo_settings()

    result = await build_client(HTTPClientManager(), settings).search(
        "today's weather in Taipei"
    )

    assert result.answer
    assert result.resources


async def test_agent_loop_with_websearch_live() -> None:
    settings = require_felo_settings()
    api_key = require_openai_api_key()

    http = HTTPClientManager()
    registry = ToolRegistry()
    client = build_client(http, settings)
    registry.register(WebSearchTool, factory=lambda: WebSearchTool(client))
    llm = OpenAIAdapter(
        ProviderConfig(
            name='openai',
            adapter='openai',
            api_key=api_key,
            models=(_OPENAI_MODEL,),
        ),
        http,
    )
    turn = TurnInput(
        instructions=(
            'Use the websearch tool to find current information before answering.'
        ),
        user_input="What is today's weather in Taipei? Answer briefly.",
        model=_OPENAI_MODEL,
    )

    fragments: list[str] = []
    output = await AgentRunner(llm, registry).run(turn, fragments.append)

    executions = [record for step in output.steps for record in step.tools]

    assert output.completed
    assert output.text == ''.join(fragments)
    assert any(record.name == 'websearch' and record.ok for record in executions)
