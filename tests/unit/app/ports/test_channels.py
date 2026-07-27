from typing import override

from app.features.agents.contracts import TextSink
from app.ports.channels import ChannelReplyContract, InboundMessage


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


def inbound() -> InboundMessage:
    return InboundMessage(
        channel='ops',
        chat='42',
        event='7',
        id='31',
        text='hello',
    )


# NOTE: ty rejects an assignment to a frozen field outright, so the runtime half of
# the check is hashability -- the property that frozen buys and that a reader can lose
def test_inbound_message_is_frozen() -> None:
    message = inbound()

    assert message.channel == 'ops'
    assert message.text == 'hello'
    assert hash(message) == hash(inbound())


# NOTE: an adapter queues one of these and its feature drains it, so a round trip
# through bytes is part of the contract rather than an implementation detail
def test_inbound_message_survives_a_round_trip_through_bytes() -> None:
    message = inbound()

    restored = InboundMessage.model_validate_json(message.model_dump_json())

    assert restored == message
