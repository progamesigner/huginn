from app.contracts.channels import (
    InboundAttachment,
    InboundCommand,
    InboundMessage,
)
from app.features.agents.contracts import TurnStream
from tests.helpers.replies import RecordingReply


# NOTE: typing the parameter is the assertion — it makes ty prove a core reply contract
# satisfies the loop's stream without either side importing the other
async def drive(sink: TurnStream, agent: str, *fragments: str) -> None:
    await sink.on_routed(agent)
    for fragment in fragments:
        await sink.on_text(fragment)
    await sink.on_tool('websearch', 'started')
    await sink.on_tool('websearch', 'finished')


async def test_reply_port_is_usable_as_the_agent_turn_stream() -> None:
    reply = RecordingReply()

    await drive(reply, 'scout', 'He', 'llo')
    await reply.finish()

    assert reply.routed == ['scout']
    assert reply.fragments == ['He', 'llo']
    assert reply.tools == [('websearch', 'started'), ('websearch', 'finished')]
    assert reply.finished
    assert not reply.abandoned


def inbound() -> InboundMessage:
    return InboundMessage(
        channel='ops',
        chat='42',
        event='7',
        id='31',
        kind='telegram',
        text='hello',
        attachments=(
            InboundAttachment(
                kind='image',
                mime='image/jpeg',
                reference='photo-file',
                unique='photo-unique',
                height=960,
                size=400_000,
                width=1280,
            ),
        ),
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
    assert restored.attachments[0].reference == 'photo-file'


# NOTE: the queue only ever carries the reference, so bytes staying out of the contract is
# the property that keeps a coalesced round small enough to sit in redis
def test_inbound_attachment_carries_a_reference_rather_than_content() -> None:
    fields = set(InboundAttachment.model_fields)

    assert 'data' not in fields
    assert 'content' not in fields
    assert 'reference' in fields


# NOTE: the mailbox is where a command waits its turn behind the round in progress, so
# it has to survive the queue as a command rather than as text to re-parse
def test_inbound_command_survives_a_round_trip_through_bytes() -> None:
    message = inbound().model_copy(
        update={'command': InboundCommand(name='register', args='ABC123')},
    )

    restored = InboundMessage.model_validate_json(message.model_dump_json())

    assert restored.command == InboundCommand(name='register', args='ABC123')
