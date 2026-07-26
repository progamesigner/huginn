from collections.abc import AsyncIterator, Sequence
from typing import override

import pytest

from app.features.agents.config import AgentConfig
from app.features.agents.exceptions import UnroutableError
from app.features.agents.router import Router
from app.features.agents.types import RoutingPolicy, SelectionPolicy
from lib.llm import (
    CompletionChunk,
    CompletionRequest,
    LLMClient,
    LLMContract,
    ProviderConfig,
    TextChunk,
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

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client
        self.calls = 0

    @override
    def client_for(self, model: str) -> LLMClient | None:
        self.calls += 1
        return self.client

    @override
    def provider_for(self, model: str) -> ProviderConfig | None:
        return None


def _agents() -> dict[str, AgentConfig]:
    return {
        'scout': AgentConfig(description='Research agent', instructions='be scout'),
        'concierge': AgentConfig(description='Fallback agent', instructions='be nice'),
    }


def _selection(model: str | None = None) -> SelectionPolicy:
    return SelectionPolicy(fallback='concierge', model=model)


def _saying(text: str) -> FakeClient:
    return FakeClient([TextChunk(text=text)])


async def test_resolve_returns_bound_agent_without_calling_the_llm() -> None:
    agents = _agents()
    llm = FakeLLM(_saying('scout'))
    policy = RoutingPolicy(bindings={'123': 'scout'}, selection=_selection())

    agent = await Router(agents, llm, 'default-model').resolve(policy, '123', 'hello')

    assert agent is agents['scout']
    assert llm.calls == 0


async def test_resolve_uses_router_llm_choice_when_valid() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(_saying('scout')), 'm').resolve(
        policy,
        '999',
        'research this',
    )

    assert agent is agents['scout']


async def test_resolve_parses_choice_from_a_decorated_reply() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(_saying('I choose scout.')), 'm').resolve(
        policy,
        '999',
        'hi',
    )

    assert agent is agents['scout']


async def test_resolve_falls_back_when_llm_picks_unknown_agent() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(_saying('banana')), 'm').resolve(
        policy,
        '999',
        'hi',
    )

    assert agent is agents['concierge']


async def test_resolve_falls_back_when_reply_is_ambiguous() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(_saying('scout or concierge')), 'm').resolve(
        policy,
        '999',
        'hi',
    )

    assert agent is agents['concierge']


async def test_resolve_falls_back_when_llm_call_errors() -> None:
    agents = _agents()
    client = FakeClient(error=TimeoutError('slow'))
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(client), 'm').resolve(policy, '999', 'hi')

    assert agent is agents['concierge']


async def test_resolve_falls_back_when_router_model_has_no_provider() -> None:
    agents = _agents()
    policy = RoutingPolicy(selection=_selection())

    agent = await Router(agents, FakeLLM(None), 'm').resolve(policy, '999', 'hi')

    assert agent is agents['concierge']


async def test_resolve_uses_router_model_for_selection() -> None:
    client = _saying('scout')
    policy = RoutingPolicy(selection=_selection(model='router-model'))

    await Router(_agents(), FakeLLM(client), 'default-model').resolve(
        policy,
        '999',
        'hi',
    )

    assert client.requests[0].model == 'router-model'


async def test_resolve_falls_back_to_default_model_for_selection() -> None:
    client = _saying('scout')
    policy = RoutingPolicy(selection=_selection())

    await Router(_agents(), FakeLLM(client), 'default-model').resolve(
        policy,
        '999',
        'hi',
    )

    assert client.requests[0].model == 'default-model'


async def test_resolve_raises_when_unbound_and_no_selection_policy() -> None:
    agents = _agents()
    policy = RoutingPolicy(bindings={'123': 'scout'})

    with pytest.raises(UnroutableError, match='999'):
        await Router(agents, FakeLLM()).resolve(policy, '999', 'hello')
