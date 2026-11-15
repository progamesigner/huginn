from typing import Annotated

import pytest
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from lib.http import HTTPClientManager
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    FinishChunk,
    Message,
    TextSegment,
)
from lib.llm.adapters.anthropic import AnthropicAdapter
from lib.llm.adapters.gemini import GeminiAdapter
from lib.llm.adapters.openai import OpenAIAdapter
from lib.llm.client import LLMClient, ProviderConfig

pytestmark = pytest.mark.integration

_OPENAI_MODEL = 'gpt-5-mini'
_ANTHROPIC_MODEL = 'claude-haiku-4-5-20251001'

_WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': "Get a city's current weather",
        'parameters': {
            'type': 'object',
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
        },
    },
}


class LiveCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    openai_api_key: Annotated[SecretStr | None, Field()] = None
    gemini_api_key: Annotated[SecretStr | None, Field()] = None
    anthropic_api_key: Annotated[SecretStr | None, Field()] = None


def require_openai_api_key() -> str:
    key = LiveCredentials().openai_api_key
    if key is None:
        pytest.skip('OPENAI_API_KEY not set')
    return key.get_secret_value()


def require_anthropic_api_key() -> str:
    key = LiveCredentials().anthropic_api_key
    if key is None:
        pytest.skip('ANTHROPIC_API_KEY not set')
    return key.get_secret_value()


async def complete(
    client: LLMClient,
    request: CompletionRequest,
) -> tuple[CompletionChunk, ...]:
    return tuple([chunk async for chunk in client.stream(request)])


async def test_openai_compat_completion() -> None:
    api_key = require_openai_api_key()
    request = CompletionRequest(
        model=_OPENAI_MODEL,
        messages=(
            Message(
                role='user',
                segments=(TextSegment(text='Reply with the single word: pong'),),
            ),
        ),
    )

    chunks = await complete(
        OpenAIAdapter(
            ProviderConfig(
                name='openai',
                adapter='openai',
                api_key=SecretStr(api_key),
                models=(_OPENAI_MODEL,),
            ),
            HTTPClientManager(),
        ),
        request,
    )

    assert any(isinstance(chunk, FinishChunk) for chunk in chunks)


async def test_openai_compat_completion_with_tools() -> None:
    api_key = require_openai_api_key()
    request = CompletionRequest(
        model=_OPENAI_MODEL,
        messages=(
            Message(
                role='user',
                segments=(TextSegment(text='Use the get_weather tool for Taipei.'),),
            ),
        ),
        tools=(_WEATHER_TOOL,),
    )

    chunks = await complete(
        OpenAIAdapter(
            ProviderConfig(
                name='openai',
                adapter='openai',
                api_key=SecretStr(api_key),
                models=(_OPENAI_MODEL,),
            ),
            HTTPClientManager(),
        ),
        request,
    )

    assert any(isinstance(chunk, FinishChunk) for chunk in chunks)


async def test_anthropic_native_completion() -> None:
    api_key = require_anthropic_api_key()
    request = CompletionRequest(
        model=_ANTHROPIC_MODEL,
        messages=(
            Message(
                role='user',
                segments=(TextSegment(text='Reply with the single word: pong'),),
            ),
        ),
    )

    chunks = await complete(
        AnthropicAdapter(
            ProviderConfig(
                name='anthropic',
                adapter='anthropic',
                api_key=SecretStr(api_key),
                models=(_ANTHROPIC_MODEL,),
            ),
            HTTPClientManager(),
        ),
        request,
    )

    assert any(isinstance(chunk, FinishChunk) for chunk in chunks)


async def test_anthropic_native_completion_with_tools() -> None:
    api_key = require_anthropic_api_key()
    request = CompletionRequest(
        model=_ANTHROPIC_MODEL,
        messages=(
            Message(
                role='user',
                segments=(TextSegment(text='Use the get_weather tool for Taipei.'),),
            ),
        ),
        tools=(_WEATHER_TOOL,),
    )

    chunks = await complete(
        AnthropicAdapter(
            ProviderConfig(
                name='anthropic',
                adapter='anthropic',
                api_key=SecretStr(api_key),
                models=(_ANTHROPIC_MODEL,),
            ),
            HTTPClientManager(),
        ),
        request,
    )

    assert any(isinstance(chunk, FinishChunk) for chunk in chunks)


async def test_gemini_native_completion() -> None:
    key = LiveCredentials().gemini_api_key
    if key is None:
        pytest.skip('GEMINI_API_KEY not set')
    http = HTTPClientManager()
    try:
        chunks = await complete(
            GeminiAdapter(
                ProviderConfig(
                    name='google',
                    adapter='gemini',
                    api_key=key,
                    models=('gemini-3.8-flash',),
                ),
                http,
            ),
            CompletionRequest(
                model='gemini-3.8-flash',
                messages=(
                    Message(
                        role='user',
                        segments=(
                            TextSegment(text='Reply with the single word: pong'),
                        ),
                    ),
                ),
            ),
        )
        assert any(isinstance(chunk, FinishChunk) for chunk in chunks)
    finally:
        await http.dispose()
