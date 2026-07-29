from typing import cast

from app.features.agents.config import AgentConfig
from app.features.agents.router import Router
from app.features.channels.config import (
    ChannelConfig,
    ChannelRouterConfig,
    routing_policy,
)
from lib.llm import LLMContract


async def test_channel_instance_can_route_directly_to_its_agent() -> None:
    scout = AgentConfig(
        description='Research agent',
        instructions='You are scout.',
    )
    router = Router(
        agents={'scout': scout},
        llm=cast(LLMContract, object()),
    )

    resolved = await router.resolve(
        routing_policy(ChannelConfig(kind='telegram', agent='scout')),
        chat_id='42',
        message='hello',
    )

    assert resolved.config is scout


def testrouting_policy_carries_bindings_and_default_agent() -> None:
    channel = ChannelConfig(
        kind='telegram',
        agent='concierge',
        bindings={'42': 'scout'},
    )

    policy = routing_policy(channel)

    assert policy.bindings == {'42': 'scout'}
    assert policy.agent == 'concierge'


def testrouting_policy_maps_the_channel_router_onto_a_selection() -> None:
    channel = ChannelConfig(
        kind='telegram',
        router=ChannelRouterConfig(fallback='concierge', model='gpt-5-mini'),
    )

    selection = routing_policy(channel).selection

    assert selection is not None
    assert selection.fallback == 'concierge'
    assert selection.model == 'gpt-5-mini'


def testrouting_policy_leaves_selection_unset_without_a_channel_router() -> None:
    policy = routing_policy(ChannelConfig(kind='telegram', agent='scout'))

    assert policy.selection is None
