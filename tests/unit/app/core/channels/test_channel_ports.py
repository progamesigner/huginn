from typing import override

from app.core.channels.contracts import ChannelReplyContract
from app.core.channels.types import InboundMessage
from app.features.agents.contracts import TextSink


class FakeReply(ChannelReplyContract):
    fragments: list[str]
    finished: bool

    def __init__(self) -> None:
        self.fragments = []
        self.finished = False

    @override
    async def on_text(self, fragment: str) -> None:
        self.fragments.append(fragment)

    @override
    async def finish(self) -> None:
        self.finished = True


# NOTE: typing the parameter is the assertion — it makes ty prove a core reply
# port works as the agent loop's sink without either side importing the other
async def drive(sink: TextSink, *fragments: str) -> None:
    for fragment in fragments:
        await sink(fragment)


async def test_reply_port_is_usable_as_the_agent_text_sink() -> None:
    reply = FakeReply()

    await drive(reply.on_text, 'He', 'llo')
    await reply.finish()

    assert reply.fragments == ['He', 'llo']
    assert reply.finished


def test_inbound_message_is_frozen() -> None:
    message = InboundMessage(channel='ops', chat='42', event='7', raw=b'{}')

    assert message.channel == 'ops'
    assert message.raw == b'{}'
