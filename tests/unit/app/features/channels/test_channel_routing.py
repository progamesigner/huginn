from typing import cast

from app.features.agents.config import AgentConfig, RouterConfig
from app.features.agents.router import Router
from app.features.channels.config import (
    ChannelConfig,
    routing_policy,
)
from lib.llm import LLMContract
from tests.helpers.logging import make_logger


async def test_channel_instance_can_route_directly_to_its_agent() -> None:
    scout = AgentConfig(
        description='Research agent',
        instructions='You are scout.',
    )
    router = Router(
        agents={'scout': scout},
        llm=cast(LLMContract, object()),
        logger=make_logger(),
    )

    resolved = await router.resolve(
        routing_policy(ChannelConfig(kind='telegram', agent='scout')),
        chat_id='42',
        message='hello',
    )

    assert resolved.config is scout


def test_routing_policy_carries_authorization_and_pinned_agent() -> None:
    channel = ChannelConfig(
        kind='telegram',
        agent='concierge',
        agents=('scout', 'concierge'),
    )

    policy = routing_policy(channel)

    assert policy.agent == 'concierge'
    assert policy.allowed == frozenset({'scout', 'concierge'})


def testrouting_policy_maps_the_channel_router_onto_a_selection() -> None:
    channel = ChannelConfig(
        kind='telegram',
        router=RouterConfig(
            fallback='concierge',
            allowed=('scout', 'concierge'),
            model='gpt-5-mini',
        ),
    )

    selection = routing_policy(channel).selection

    assert selection is not None
    assert selection.fallback == 'concierge'
    assert selection.model == 'gpt-5-mini'
    assert selection.allowed == frozenset({'scout', 'concierge'})


def testrouting_policy_leaves_selection_unset_without_a_channel_router() -> None:
    policy = routing_policy(ChannelConfig(kind='telegram', agent='scout'))

    assert policy.selection is None
