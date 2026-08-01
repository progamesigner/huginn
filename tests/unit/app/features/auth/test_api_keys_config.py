import re

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config.registry import HuginnConfig
from app.features.agents.config import (
    AGENTS,
    DEFAULTS,
    AgentConfig,
    AgentConfigList,
    DefaultsConfig,
)
from app.features.agents.exceptions import (
    MissingModelError,
    UnknownAgentReferenceError,
)
from app.features.auth.config import (
    API_KEYS,
    ApiKeyConfig,
    ApiKeyConfigList,
    ApiKeyRouterConfig,
    ApiKeyToolsConfig,
)
from app.features.auth.exceptions import (
    DuplicateApiKeyTokenError,
    EmptyApiKeyTokenError,
    MissingApiKeyAgentsError,
)


def key(
    credential: str = 'web-key',
    *,
    agents: tuple[str, ...] = ('scout', 'concierge'),
    agent: str | None = None,
    router: ApiKeyRouterConfig | None = None,
    tools: ApiKeyToolsConfig | None = None,
) -> ApiKeyConfig:
    return ApiKeyConfig(
        token=SecretStr(credential),
        agents=agents,
        agent=agent,
        router=router,
        tools=tools,
    )


def make_config(
    keys: dict[str, ApiKeyConfig],
    *,
    agents: tuple[str, ...] = ('scout', 'concierge'),
    model: str | None = 'gpt-5-mini',
) -> HuginnConfig:
    return HuginnConfig(
        sections={
            AGENTS.key: AgentConfigList(
                {name: AgentConfig(description=name) for name in agents},
            ),
            API_KEYS.key: ApiKeyConfigList(keys),
            DEFAULTS.key: DefaultsConfig(model=model),
        },
    )


def validate(config: HuginnConfig) -> None:
    assert API_KEYS.validate is not None

    API_KEYS.validate(config)


def describe(config: HuginnConfig) -> list[str]:
    assert API_KEYS.describe is not None

    return API_KEYS.describe(config)


def test_an_absent_section_leaves_the_api_closed_rather_than_open() -> None:
    # NOTE: the loader validates a missing section from {}, so "no keys configured" has
    # to be a legal state -- one where every request is unauthorized
    config = make_config({})

    validate(config)

    assert config.section(API_KEYS).root == {}


def test_a_key_carries_its_agents_and_optional_knobs() -> None:
    configured = key(
        agent='scout',
        router=ApiKeyRouterConfig(fallback='concierge', model='gpt-5-mini'),
        tools=ApiKeyToolsConfig(deny=('dangerous-tool',)),
    )
    config = make_config({'web-frontend': configured})

    validate(config)

    assert configured.agents == ('scout', 'concierge')
    assert configured.agent == 'scout'
    assert configured.router == ApiKeyRouterConfig(
        fallback='concierge',
        model='gpt-5-mini',
    )
    assert configured.tools == ApiKeyToolsConfig(deny=('dangerous-tool',))


def test_a_key_keeps_its_token_out_of_a_dump() -> None:
    keys = ApiKeyConfigList({'web-frontend': key('secret-token')})

    assert 'secret-token' not in keys.model_dump_json()


def test_a_key_requires_a_token() -> None:
    with pytest.raises(ValidationError, match='token'):
        ApiKeyConfig.model_validate_json('{"agents": ["scout"]}')


def test_a_key_rejects_an_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ApiKeyConfig.model_validate_json(
            '{"token": "x", "agents": ["scout"], "bogus": true}',
        )


def test_an_empty_token_is_rejected_at_load() -> None:
    # NOTE: an unset environment variable resolves to '' -- without this check the key
    # would silently authenticate a caller who sent 'Bearer '
    config = make_config({'web-frontend': key('')})

    with pytest.raises(
        EmptyApiKeyTokenError,
        match=re.escape('api_keys.web-frontend.token'),
    ):
        validate(config)


def test_two_keys_may_not_share_a_token() -> None:
    config = make_config(
        {'web-frontend': key('shared'), 'mobile-app': key('shared')},
    )

    with pytest.raises(DuplicateApiKeyTokenError, match='duplicates api_keys'):
        validate(config)


def test_a_key_with_no_agents_is_rejected() -> None:
    config = make_config({'web-frontend': key(agents=())})

    with pytest.raises(MissingApiKeyAgentsError, match='at least one agent'):
        validate(config)


def test_a_key_may_only_name_configured_agents() -> None:
    config = make_config({'web-frontend': key(agents=('scout', 'ghost'))})

    with pytest.raises(UnknownAgentReferenceError, match=re.escape('agents[1]')):
        validate(config)


def test_the_pinned_agent_must_be_in_the_allowed_list() -> None:
    config = make_config(
        {'web-frontend': key(agents=('concierge',), agent='scout')},
    )

    with pytest.raises(UnknownAgentReferenceError, match='is not in'):
        validate(config)


def test_the_router_fallback_must_be_in_the_allowed_list() -> None:
    config = make_config(
        {
            'web-frontend': key(
                agents=('scout',),
                router=ApiKeyRouterConfig(fallback='concierge'),
            ),
        },
    )

    with pytest.raises(UnknownAgentReferenceError, match=re.escape('router.fallback')):
        validate(config)


def test_a_router_without_a_model_anywhere_is_rejected() -> None:
    config = make_config(
        {'web-frontend': key(router=ApiKeyRouterConfig(fallback='concierge'))},
        model=None,
    )

    with pytest.raises(MissingModelError, match=re.escape('router.model')):
        validate(config)


def test_the_description_names_the_key_and_its_agents_but_no_token() -> None:
    config = make_config({'web-frontend': key('secret-token')})

    described = describe(config)

    assert described == ['api_keys: web-frontend (scout, concierge)']
    assert 'secret-token' not in described[0]


def test_the_description_of_an_empty_section_is_silent() -> None:
    assert describe(make_config({})) == []
