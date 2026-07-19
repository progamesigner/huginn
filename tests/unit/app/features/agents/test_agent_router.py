import pytest

from app.features.agents.config import AgentConfig, ChannelConfig, ChannelRouterConfig
from app.features.agents.exceptions import UnroutableError
from app.features.agents.router import Router


def _agents() -> dict[str, AgentConfig]:
    return {
        'scout': AgentConfig(description='Research agent', instructions='be scout'),
        'concierge': AgentConfig(description='Fallback agent', instructions='be nice'),
    }


async def test_resolve_returns_bound_agent_for_known_chat() -> None:
    agents = _agents()
    channel = ChannelConfig(
        bindings={'123': 'scout'},
        router=ChannelRouterConfig(fallback='concierge'),
    )

    agent = await Router(agents).resolve(channel, '123', 'hello')

    assert agent is agents['scout']


async def test_resolve_uses_fallback_when_chat_is_unbound() -> None:
    agents = _agents()
    channel = ChannelConfig(
        bindings={'123': 'scout'},
        router=ChannelRouterConfig(fallback='concierge'),
    )

    agent = await Router(agents).resolve(channel, '999', 'hello')

    assert agent is agents['concierge']


async def test_resolve_raises_when_unbound_and_channel_has_no_router() -> None:
    agents = _agents()
    channel = ChannelConfig(bindings={'123': 'scout'})

    with pytest.raises(UnroutableError, match='999'):
        await Router(agents).resolve(channel, '999', 'hello')
