from typing import Annotated

import pytest
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from lib.decision import (
    ChoiceQuestion,
    DecisionManager,
    DecisionProviderConfig,
    DecisionRef,
    DecisionRequest,
)
from lib.http import HTTPClientManager
from tests.helpers.logging import make_logger

pytestmark = pytest.mark.integration

_TYPESAFE_MODEL = 'jev-1.13.0'
_OPENROUTER_MODEL = 'typesafe/jev-1.13'
_OPENROUTER_URL = 'https://openrouter.ai/api/alpha/decisions'
_OPENAI_MODEL = 'gpt-6-luna'
_BIFROST_MODEL = 'typesafe/jev-1.13.0'

_CRITERIA = {
    'weather': 'Questions about weather and forecasts.',
    'billing': 'Charges, invoices and refunds.',
    '__none__': 'None of the options fit.',
}


class LiveCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    typesafe_api_key: Annotated[SecretStr | None, Field()] = None
    openrouter_api_key: Annotated[SecretStr | None, Field()] = None
    openai_api_key: Annotated[SecretStr | None, Field()] = None
    gemini_api_key: Annotated[SecretStr | None, Field()] = None
    bifrost_api_key: Annotated[SecretStr | None, Field()] = None
    bifrost_base_url: Annotated[str | None, Field()] = None


def _provider(name: str) -> tuple[DecisionProviderConfig, str]:
    credentials = LiveCredentials()
    match name:
        case 'gemini' if credentials.gemini_api_key is not None:
            return DecisionProviderConfig(
                name=name,
                adapter='gemini',
                api_key=credentials.gemini_api_key,
                models=('gemini-decision-1',),
            ), 'gemini-decision-1'
        case 'typesafe' if credentials.typesafe_api_key is not None:
            return DecisionProviderConfig(
                name=name,
                adapter='typesafe',
                api_key=credentials.typesafe_api_key,
                models=(_TYPESAFE_MODEL,),
            ), _TYPESAFE_MODEL
        case 'openrouter' if credentials.openrouter_api_key is not None:
            return DecisionProviderConfig(
                name=name,
                adapter='typesafe',
                api_key=credentials.openrouter_api_key,
                models=(_OPENROUTER_MODEL,),
                url=_OPENROUTER_URL,
            ), _OPENROUTER_MODEL
        case 'openai' if credentials.openai_api_key is not None:
            return DecisionProviderConfig(
                name=name,
                adapter='openai',
                api_key=credentials.openai_api_key,
                models=(_OPENAI_MODEL,),
            ), _OPENAI_MODEL
        case 'bifrost' if (
            credentials.bifrost_api_key is not None
            and credentials.bifrost_base_url is not None
        ):
            return DecisionProviderConfig(
                name=name,
                adapter='bifrost',
                api_key=credentials.bifrost_api_key,
                models=(_BIFROST_MODEL,),
                url=f'{credentials.bifrost_base_url.rstrip("/")}/v1/decisions',
            ), _BIFROST_MODEL
        case 'bifrost-openai' if (
            credentials.bifrost_api_key is not None
            and credentials.bifrost_base_url is not None
        ):
            return DecisionProviderConfig(
                name=name,
                adapter='openai',
                api_key=credentials.bifrost_api_key,
                models=(_BIFROST_MODEL,),
                url=(f'{credentials.bifrost_base_url.rstrip("/")}/openai/v1/decisions'),
            ), _BIFROST_MODEL
        case _:
            pytest.skip(f'credentials for {name} not set')


@pytest.mark.parametrize(
    'name',
    ['typesafe', 'openrouter', 'openai', 'bifrost', 'bifrost-openai', 'gemini'],
)
async def test_live_choice_picks_one_of_the_offered_options(name: str) -> None:
    provider, model = _provider(name)
    client = DecisionManager(
        (provider,),
        HTTPClientManager(),
        logger=make_logger(),
    ).client_for(DecisionRef(model=model))
    assert client is not None

    result = await client.decide(
        DecisionRequest(
            model=model,
            state={
                'current': ['I was charged twice for last month, please refund one'],
            },
            questions={
                'route': ChoiceQuestion(
                    instructions='Which team should handle the current message?',
                    criteria=_CRITERIA,
                ),
            },
        ),
    )

    assert result.answers['route'].choice == 'billing'
    assert result.answers['route'].confidence is not None
