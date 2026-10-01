from pydantic import SecretStr

from app.contracts.auth import ApiRouting
from app.core.config.registry import HuginnConfig
from app.features.agents.config import RouterConfig, RouterShadowConfig
from app.features.auth.config import (
    API_KEYS,
    ApiKeyConfig,
    ApiKeyConfigList,
    ApiKeyToolsConfig,
)
from app.features.auth.services import ApiKeyDirectory
from tests.helpers.logging import make_logger


def directory(**keys: ApiKeyConfig) -> ApiKeyDirectory:
    return ApiKeyDirectory(
        HuginnConfig(sections={API_KEYS.key: ApiKeyConfigList(keys)}),
        logger=make_logger(),
    )


def test_a_known_token_resolves_to_the_client_that_carries_it() -> None:
    resolved = directory(
        web=ApiKeyConfig(
            token=SecretStr('web-token'),
            agents=('scout', 'concierge'),
            agent='scout',
            router=RouterConfig(
                fallback='concierge',
                allowed=('scout', 'concierge'),
                model='gpt-5-mini',
                min_confidence=0.5,
                min_margin=0.25,
                shadow=RouterShadowConfig(model='jev', provider='gateway'),
            ),
            tools=ApiKeyToolsConfig(deny=('dangerous-tool',)),
        ),
    ).authenticate('web-token')

    assert resolved is not None
    assert resolved.key == 'web'
    assert resolved.agents == frozenset({'scout', 'concierge'})
    assert resolved.agent == 'scout'
    assert resolved.routing == ApiRouting(
        fallback='concierge',
        model='gpt-5-mini',
        allowed=frozenset({'scout', 'concierge'}),
        min_confidence=0.5,
        min_margin=0.25,
        shadow_model='jev',
        shadow_provider='gateway',
    )
    assert resolved.denied_tools == frozenset({'dangerous-tool'})


def test_a_client_without_optional_knobs_carries_none_of_them() -> None:
    resolved = directory(
        web=ApiKeyConfig(token=SecretStr('web-token'), agents=('scout',)),
    ).authenticate('web-token')

    assert resolved is not None
    assert resolved.agent is None
    assert resolved.routing is None
    assert resolved.denied_tools == frozenset()


def test_each_token_resolves_to_its_own_client() -> None:
    resolver = directory(
        web=ApiKeyConfig(token=SecretStr('web-token'), agents=('scout',)),
        mobile=ApiKeyConfig(token=SecretStr('mobile-token'), agents=('concierge',)),
    )

    first = resolver.authenticate('web-token')
    second = resolver.authenticate('mobile-token')

    assert first is not None
    assert second is not None
    assert (first.key, second.key) == ('web', 'mobile')


def test_an_unknown_token_resolves_to_nobody() -> None:
    resolver = directory(
        web=ApiKeyConfig(token=SecretStr('web-token'), agents=('scout',)),
    )

    assert resolver.authenticate('wrong-token') is None
    assert resolver.authenticate('') is None
    assert resolver.authenticate('web-token ') is None


def test_no_configured_key_authenticates_nobody() -> None:
    assert directory().authenticate('web-token') is None
