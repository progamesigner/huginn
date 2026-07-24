from typing import cast

from app.features.agents.config import AgentConfig
from app.features.agents.router import Router
from app.features.channels.config import ChannelConfig
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
        ChannelConfig(kind='telegram', agent='scout'),
        chat_id='42',
        message='hello',
    )

    assert resolved is scout
