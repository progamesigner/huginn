from collections.abc import AsyncIterator, Sequence
from typing import override

import pytest

from app.features.agents.config import AgentConfig, DefaultsConfig
from app.features.agents.exceptions import UnroutableError
from app.features.agents.router import Router
from app.features.agents.types import RoutingPolicy, SelectionPolicy
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    LLMClient,
    LLMContract,
    ModelRef,
    ProviderConfig,
    TextChunk,
    TextSegment,
)


class FakeClient:
    chunks: list[CompletionChunk]
    error: Exception | None
    requests: list[CompletionRequest]

    def __init__(
        self,
        chunks: Sequence[CompletionChunk] = (),
        error: Exception | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.error = error
        self.requests = []

    async def stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[CompletionChunk]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error

        for chunk in self.chunks:
            yield chunk


class FakeLLM(LLMContract):
    client: LLMClient | None
    calls: int
    provider: ProviderConfig | None
    refs: list[ModelRef]

    def __init__(
        self,
        client: LLMClient | None = None,
        provider: ProviderConfig | None = None,
    ) -> None:
        self.client = client
        self.calls = 0
        self.provider = provider
        self.refs = []

    @override
    def client_for(self, ref: ModelRef) -> LLMClient | None:
        self.calls += 1
        self.refs.append(ref)
        return self.client

    @override
    def provider_for(self, ref: ModelRef) -> ProviderConfig | None:
        return self.provider


def _agents() -> dict[str, AgentConfig]:
    return {
        'scout': AgentConfig(description='Research agent', instructions='be scout'),
        'concierge': AgentConfig(description='Fallback agent', instructions='be nice'),
    }


def _defaults(model: str | None = None, provider: str | None = None) -> DefaultsConfig:
    return DefaultsConfig(model=model, provider=provider)


def _selection(
    model: str | None = None,
    provider: str | None = None,
) -> SelectionPolicy:
    return SelectionPolicy(fallback='concierge', model=model, provider=provider)


def _saying(text: str) -> FakeClient:
    return FakeClient([TextChunk(text=text)])


async def test_resolve_returns_bound_agent_without_calling_the_llm() -> None:
    agents = _agents()
    llm = FakeLLM(_saying('scout'))
    policy = RoutingPolicy(bindings={'123': 'scout'}, selection=_selection())

    agent = await Router(agents, llm, _defaults('default-model')).resolve(
        policy,
        '123',
        'hello',
    )

    assert agent.config is agents['scout']
    # NOTE: the name is half the answer -- a conversation is keyed by it, and an
    # AgentConfig cannot report the config key it was read from
    assert agent.name == 'scout'
    assert llm.calls == 0


async def test_resolve_uses_router_llm_choice_when_valid() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(_saying('scout')), _defaults('m')).resolve(
        policy,
        '999',
        'research this',
    )

    assert agent.config is agents['scout']


async def test_resolve_parses_choice_from_a_decorated_reply() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(
        agents,
        FakeLLM(_saying('I choose scout.')),
        _defaults('m'),
    ).resolve(
        policy,
        '999',
        'hi',
    )

    assert agent.config is agents['scout']


async def test_resolve_falls_back_when_llm_picks_unknown_agent() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(_saying('banana')), _defaults('m')).resolve(
        policy,
        '999',
        'hi',
    )

    assert agent.config is agents['concierge']


async def test_resolve_falls_back_when_reply_is_ambiguous() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(
        agents,
        FakeLLM(_saying('scout or concierge')),
        _defaults('m'),
    ).resolve(
        policy,
        '999',
        'hi',
    )

    assert agent.config is agents['concierge']


async def test_resolve_falls_back_when_llm_call_errors() -> None:
    agents = _agents()
    client = FakeClient(error=TimeoutError('slow'))
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(client), _defaults('m')).resolve(
        policy,
        '999',
        'hi',
    )

    assert agent.config is agents['concierge']


async def test_resolve_falls_back_when_router_model_has_no_provider() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(None), _defaults('m')).resolve(
        policy,
        '999',
        'hi',
    )

    assert agent.config is agents['concierge']


async def test_resolve_uses_router_model_for_selection() -> None:
    client = _saying('scout')
    policy = RoutingPolicy(selection=_selection(model='router-model'))

    await Router(_agents(), FakeLLM(client), _defaults('default-model')).resolve(
        policy,
        '999',
        'hi',
    )

    assert client.requests[0].model == 'router-model'


async def test_resolve_falls_back_to_default_model_for_selection() -> None:
    client = _saying('scout')
    policy = RoutingPolicy(selection=_selection())

    await Router(_agents(), FakeLLM(client), _defaults('default-model')).resolve(
        policy,
        '999',
        'hi',
    )

    assert client.requests[0].model == 'default-model'


async def test_resolve_pins_the_router_llm_to_the_configured_provider() -> None:
    llm = FakeLLM(_saying('scout'))
    policy = RoutingPolicy(
        selection=_selection(model='claude-sonnet-5', provider='bifrost'),
    )

    await Router(_agents(), llm, _defaults('default-model')).resolve(
        policy,
        '999',
        'hi',
    )

    assert llm.refs == [ModelRef(model='claude-sonnet-5', provider='bifrost')]


async def test_resolve_leaves_the_router_provider_unpinned_when_only_a_model_is_set() -> (
    None
):
    # NOTE: defaults.provider backs defaults.model only -- naming a router model on
    # its own means "whichever provider offers it"
    llm = FakeLLM(_saying('scout'))
    policy = RoutingPolicy(selection=_selection(model='router-model'))
    defaults = _defaults('default-model', provider='openai')

    await Router(_agents(), llm, defaults).resolve(policy, '999', 'hi')

    assert llm.refs == [ModelRef(model='router-model')]


async def test_resolve_inherits_the_default_provider_with_the_default_model() -> None:
    llm = FakeLLM(_saying('scout'))
    policy = RoutingPolicy(selection=_selection())
    defaults = _defaults('default-model', provider='openai')

    await Router(_agents(), llm, defaults).resolve(policy, '999', 'hi')

    assert llm.refs == [ModelRef(model='default-model', provider='openai')]


async def test_resolve_raises_when_unbound_and_no_selection_policy() -> None:
    agents = _agents()
    policy = RoutingPolicy(bindings={'123': 'scout'})

    with pytest.raises(UnroutableError, match='999'):
        await Router(agents, FakeLLM()).resolve(policy, '999', 'hello')


async def test_resolve_offers_the_router_only_the_agents_the_caller_may_reach() -> None:
    # NOTE: narrowing the pool is what makes an out-of-permission choice impossible
    # rather than merely rejected -- the model never sees the agent it may not pick
    client = _saying('concierge')
    policy = RoutingPolicy(
        selection=_selection(),
        allowed=frozenset({'concierge'}),
    )

    routed = await Router(
        _agents(),
        FakeLLM(client),
        _defaults('default-model'),
    ).resolve(
        policy,
        '999',
        'hello',
    )

    prompt = client.requests[0].messages[0].segments[0]
    assert isinstance(prompt, TextSegment)
    assert 'concierge' in prompt.text
    assert 'scout' not in prompt.text
    assert routed.name == 'concierge'


async def test_resolve_refuses_a_binding_outside_what_the_caller_may_reach() -> None:
    policy = RoutingPolicy(
        bindings={'123': 'scout'},
        allowed=frozenset({'concierge'}),
    )

    with pytest.raises(UnroutableError, match='scout'):
        await Router(_agents(), FakeLLM()).resolve(policy, '123', 'hello')


async def test_resolve_refuses_a_pinned_agent_outside_what_the_caller_may_reach() -> (
    None
):
    policy = RoutingPolicy(agent='scout', allowed=frozenset({'concierge'}))

    with pytest.raises(UnroutableError, match='scout'):
        await Router(_agents(), FakeLLM()).resolve(policy, '999', 'hello')


async def test_resolve_leaves_every_agent_routable_when_nothing_narrows_it() -> None:
    # NOTE: a channel whose config already names its agents passes no permission, and
    # must keep reaching all of them
    policy = RoutingPolicy(agent='scout')

    routed = await Router(_agents(), FakeLLM()).resolve(policy, '999', 'hello')

    assert routed.name == 'scout'


async def test_resolve_falls_back_to_the_default_preamble_without_configuration() -> (
    None
):
    client = _saying('scout')
    policy = RoutingPolicy(selection=_selection())

    await Router(_agents(), FakeLLM(client), _defaults('m')).resolve(
        policy,
        '999',
        'hi',
    )

    prompt = client.requests[0].messages[0].segments[0]
    assert isinstance(prompt, TextSegment)
    assert prompt.text.startswith('You route a message to exactly one agent.')


async def test_resolve_uses_a_configured_router_preamble() -> None:
    client = _saying('scout')
    policy = RoutingPolicy(selection=_selection())
    router = Router(
        _agents(),
        FakeLLM(client),
        _defaults('m'),
        prompt='Pick the best agent.',
    )

    await router.resolve(policy, '999', 'hi')

    prompt = client.requests[0].messages[0].segments[0]
    assert isinstance(prompt, TextSegment)
    # NOTE: the configured preamble replaces the default wording, but the agent
    # roster is always appended -- _parse_choice depends on it being there
    assert prompt.text.startswith('Pick the best agent.')
    assert 'Agents:' in prompt.text
    assert '- scout: Research agent' in prompt.text
