from typing import override
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.contracts.channels import (
    ChannelGatewayContract,
    ChannelReplyContract,
    InboundAttachment,
    InboundCommand,
    InboundMessage,
)
from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.types import TurnRequest
from app.features.channels.contracts import TranscriptionContract
from app.features.channels.exceptions import TranscriptionUnavailableError
from app.features.channels.registry import CommandRegistry
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.channels.telegram.turns import TelegramTurnProcessor
from app.features.identities.contracts import IdentityResolverContract
from app.features.identities.types import (
    AnonymousPrincipal,
    LinkedPrincipal,
    Principal,
)
from tests.helpers.replies import RecordingReply

_LINKED = UUID('00000000-0000-7000-8000-000000000001')


class FakeGateway(ChannelGatewayContract):
    abandoned: list[str]
    acknowledged: list[str]
    fetched: list[str]
    replies: list[RecordingReply]

    def __init__(self) -> None:
        self.abandoned = []
        self.acknowledged = []
        self.fetched = []
        self.replies = []

    @override
    async def abandon(self, message: InboundMessage) -> None:
        self.abandoned.append(message.id)

    @override
    async def acknowledge(self, message: InboundMessage) -> None:
        self.acknowledged.append(message.id)

    @override
    async def fetch(self, attachment: InboundAttachment) -> bytes:
        self.fetched.append(attachment.reference)
        return b'bytes'

    @override
    def reply(self, message: InboundMessage) -> ChannelReplyContract:
        reply = RecordingReply()
        self.replies.append(reply)
        return reply


class FakeTranscription(TranscriptionContract):
    error: Exception | None
    heard: list[tuple[str, bytes]]
    text: str

    def __init__(
        self,
        *,
        error: Exception | None = None,
        text: str = 'buy milk',
    ) -> None:
        self.error = error
        self.heard = []
        self.text = text

    @override
    async def transcribe(self, mime: str, data: bytes) -> str:
        self.heard.append((mime, data))
        if self.error is not None:
            raise self.error

        return self.text


class FakeIdentities(IdentityResolverContract):
    calls: list[tuple[str, str, str]]
    linked: frozenset[str]

    def __init__(self, linked: frozenset[str] = frozenset()) -> None:
        self.calls = []
        self.linked = linked

    @override
    async def resolve(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Principal:
        self.calls.append((provider, realm, external_user_id))
        if external_user_id in self.linked:
            return LinkedPrincipal(_LINKED, _LINKED)

        return AnonymousPrincipal(provider, realm, external_user_id)


class FakeTurns(TurnHandlerContract):
    error: Exception | None
    requests: list[TurnRequest]

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.requests = []

    @override
    async def handle(
        self,
        request: TurnRequest,
        reply: ChannelReplyContract,
    ) -> None:
        self.requests.append(request)
        if self.error is not None:
            raise self.error

        await reply.on_text('answer')
        await reply.finish()


def channel() -> TelegramChannelConfig:
    return TelegramChannelConfig(
        kind='telegram',
        agent='scout',
        token=SecretStr('bot-token'),
        secret_token=SecretStr('sekret'),
    )


def photo() -> InboundAttachment:
    return InboundAttachment(
        kind='image',
        mime='image/jpeg',
        reference='file-1',
        unique='unique-1',
        height=960,
        size=2048,
        width=1280,
    )


def voice() -> InboundAttachment:
    return InboundAttachment(
        kind='audio',
        mime='audio/ogg',
        reference='file-2',
        unique='unique-2',
        duration=12,
    )


def inbound(
    event: int,
    *,
    text: str = 'hello',
    user: str | None = '77',
    attachments: tuple[InboundAttachment, ...] = (),
    command: InboundCommand | None = None,
) -> InboundMessage:
    return InboundMessage(
        channel='support_bot',
        chat='4242',
        event=str(event),
        id=str(event),
        kind='telegram',
        text=text,
        attachments=attachments,
        command=command,
        user=user,
    )


def commanded(event: int, name: str, args: str = '') -> InboundMessage:
    return inbound(
        event,
        text=f'/{name} {args}'.strip(),
        command=InboundCommand(name=name, args=args),
    )


def registry(answer: str = 'done') -> CommandRegistry:
    async def handle(command: InboundCommand, message: InboundMessage) -> str:
        return f'{answer}:{command.name}'

    built = CommandRegistry()
    built.register('new', 'Start over.', handle)

    return built


def make_processor(
    turns: FakeTurns,
    identities: FakeIdentities,
    gateway: FakeGateway,
    transcription: FakeTranscription | None = None,
    commands: CommandRegistry | None = None,
) -> TelegramTurnProcessor:
    return TelegramTurnProcessor(
        turns=turns,
        identities=identities,
        gateway=lambda _: gateway,
        transcription=transcription or FakeTranscription(),
        commands=commands or registry(),
    )


async def test_a_round_answers_through_the_reply_of_its_last_message() -> None:
    gateway = FakeGateway()
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), gateway).process(
        channel(),
        (inbound(1), inbound(2)),
    )

    assert gateway.replies[0].fragments == ['answer']
    assert gateway.replies[0].finished
    assert not gateway.replies[0].abandoned


async def test_every_coalesced_message_stays_its_own_prompt() -> None:
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), FakeGateway()).process(
        channel(),
        (inbound(1, text='first'), inbound(2, text='second')),
    )

    request = turns.requests[0]
    assert [prompt.text for prompt in request.prompts] == ['first', 'second']
    assert request.chat == '4242'
    assert request.policy.agent == 'scout'


async def test_a_prompt_carries_the_resolved_principal_of_its_sender() -> None:
    identities = FakeIdentities(linked=frozenset({'77'}))
    turns = FakeTurns()

    await make_processor(turns, identities, FakeGateway()).process(
        channel(),
        (inbound(1), inbound(2, user='88')),
    )

    actors = [prompt.actor for prompt in turns.requests[0].prompts]
    assert actors == [
        f'user:{_LINKED}',
        AnonymousPrincipal('telegram', 'support_bot', '88').stable_key,
    ]
    # NOTE: the realm is the channel instance, so two bots never share a principal
    assert identities.calls == [
        ('telegram', 'support_bot', '77'),
        ('telegram', 'support_bot', '88'),
    ]


async def test_one_sender_is_resolved_once_per_round() -> None:
    identities = FakeIdentities()

    await make_processor(FakeTurns(), identities, FakeGateway()).process(
        channel(),
        (inbound(1), inbound(2), inbound(3)),
    )

    assert len(identities.calls) == 1


async def test_a_message_nobody_sent_is_recorded_unattributed() -> None:
    identities = FakeIdentities()
    turns = FakeTurns()

    await make_processor(turns, identities, FakeGateway()).process(
        channel(),
        (inbound(1, user=None),),
    )

    assert [prompt.actor for prompt in turns.requests[0].prompts] == ['']
    assert identities.calls == []


async def test_a_message_without_text_is_not_a_prompt() -> None:
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), FakeGateway()).process(
        channel(),
        (inbound(1, text=''), inbound(2, text='real')),
    )

    assert [prompt.text for prompt in turns.requests[0].prompts] == ['real']


async def test_an_attachment_reaches_the_turn_with_its_bytes_downloaded() -> None:
    gateway = FakeGateway()
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), gateway).process(
        channel(),
        (inbound(1, text='what is this?', attachments=(photo(),)),),
    )

    # NOTE: the contract carries a reference only, so the round is where the bytes are
    # fetched -- a 5MB photo through the mailbox would be a 6.7MB redis payload
    assert gateway.fetched == ['file-1']
    carried = turns.requests[0].prompts[0].attachments[0]
    assert carried.kind == 'image'
    assert carried.mime == 'image/jpeg'
    assert carried.reference == 'file-1'
    assert carried.unique == 'unique-1'
    assert carried.data == b'bytes'
    assert carried.hint == '[image: image/jpeg 1280x960]'


async def test_a_photo_without_a_caption_is_still_a_prompt() -> None:
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), FakeGateway()).process(
        channel(),
        (inbound(1, text='', attachments=(photo(),)),),
    )

    prompt = turns.requests[0].prompts[0]
    assert prompt.text == ''
    assert len(prompt.attachments) == 1


async def test_a_voice_note_is_named_by_its_duration() -> None:
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), FakeGateway()).process(
        channel(),
        (inbound(1, text='', attachments=(voice(),)),),
    )

    assert turns.requests[0].prompts[0].attachments[0].hint == '[audio: audio/ogg 12s]'


async def test_a_voice_note_carries_what_it_said() -> None:
    transcription = FakeTranscription(text='remind me at six')
    turns = FakeTurns()

    await make_processor(
        turns,
        FakeIdentities(),
        FakeGateway(),
        transcription,
    ).process(channel(), (inbound(1, text='', attachments=(voice(),)),))

    # NOTE: routing reads the prompt text, so the voice has to become words inside the
    # round rather than on the way to the model
    assert transcription.heard == [('audio/ogg', b'bytes')]
    assert turns.requests[0].prompts[0].attachments[0].transcript == 'remind me at six'


async def test_a_photo_is_never_sent_to_be_transcribed() -> None:
    transcription = FakeTranscription()

    await make_processor(
        FakeTurns(),
        FakeIdentities(),
        FakeGateway(),
        transcription,
    ).process(channel(), (inbound(1, text='', attachments=(photo(),)),))

    assert transcription.heard == []


async def test_a_voice_note_nobody_can_transcribe_leaves_the_text_answerable() -> None:
    transcription = FakeTranscription(error=RuntimeError('whisper died'))
    turns = FakeTurns()

    await make_processor(
        turns,
        FakeIdentities(),
        FakeGateway(),
        transcription,
    ).process(channel(), (inbound(1, text='and also this', attachments=(voice(),)),))

    prompt = turns.requests[0].prompts[0]
    assert prompt.text == 'and also this'
    assert prompt.attachments == ()


async def test_a_voice_only_round_says_it_could_not_read_the_voice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway = FakeGateway()
    turns = FakeTurns()
    transcription = FakeTranscription(
        error=TranscriptionUnavailableError('nothing configured'),
    )

    with caplog.at_level('WARNING'):
        await make_processor(turns, FakeIdentities(), gateway, transcription).process(
            channel(),
            (inbound(1, text='', attachments=(voice(),)),),
        )

    # NOTE: the user hears one sentence either way, so the log is the only place that
    # says whether nobody configured transcription or the vendor fell over
    assert any(
        'no transcription is configured' in record.message for record in caplog.records
    )

    # NOTE: the generic refusal names voice as something we read, so a round that was
    # only voice has to be told what actually went wrong with it
    assert turns.requests == []
    assert gateway.replies[0].fragments == [
        'I could not read that voice message; please send it as text.',
    ]
    assert gateway.replies[0].finished


async def test_an_attachment_that_will_not_download_is_left_behind() -> None:
    class RefusingGateway(FakeGateway):
        @override
        async def fetch(self, attachment: InboundAttachment) -> bytes:
            raise RuntimeError('file is over the limit')

    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), RefusingGateway()).process(
        channel(),
        (inbound(1, text='look', attachments=(photo(),)),),
    )

    # NOTE: one unreadable attachment must not cost the caption its answer
    prompt = turns.requests[0].prompts[0]
    assert prompt.text == 'look'
    assert prompt.attachments == ()


async def test_a_round_with_nothing_readable_says_so_instead_of_going_quiet() -> None:
    gateway = FakeGateway()
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), gateway).process(
        channel(),
        (inbound(1, text=''),),
    )

    # NOTE: a document or a sticker used to leave the ack reaction sitting on a message
    # that was never going to be answered -- no answer and no failure either
    assert turns.requests == []
    assert gateway.replies[0].fragments == [
        'I can only read text, photos and voice messages.',
    ]
    assert gateway.replies[0].finished
    assert gateway.abandoned == []


async def test_a_command_is_answered_without_reaching_the_loop() -> None:
    gateway = FakeGateway()
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), gateway).process(
        channel(),
        (commanded(1, 'new'),),
    )

    # NOTE: a command is the user operating the bot, not talking to it -- letting it
    # through would put '/new' in the transcript and have the model answer it
    assert turns.requests == []
    assert gateway.replies[0].fragments == ['done:new']
    assert gateway.replies[0].finished


async def test_a_command_only_round_is_not_declined_as_unreadable() -> None:
    gateway = FakeGateway()

    await make_processor(FakeTurns(), FakeIdentities(), gateway).process(
        channel(),
        (commanded(1, 'new'),),
    )

    assert len(gateway.replies) == 1


async def test_a_command_beside_text_leaves_the_text_answered() -> None:
    gateway = FakeGateway()
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), gateway).process(
        channel(),
        (inbound(1, text='what is the time'), commanded(2, 'new')),
    )

    # NOTE: each gets its own reply message: the command answers where it was typed and
    # the round still answers what was said
    assert [prompt.text for prompt in turns.requests[0].prompts] == ['what is the time']
    assert gateway.replies[0].fragments == ['done:new']
    assert gateway.replies[1].fragments == ['answer']


async def test_a_command_nobody_registered_is_reported_to_the_user() -> None:
    gateway = FakeGateway()
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), gateway).process(
        channel(),
        (commanded(1, 'reboot'),),
    )

    assert turns.requests == []
    assert gateway.replies[0].fragments == [
        '/reboot is not a command I know. Available: /new.',
    ]


async def test_a_failed_command_reports_on_its_own_message_and_nothing_else() -> None:
    async def raising(command: InboundCommand, message: InboundMessage) -> str:
        raise RuntimeError('the service is down')

    commands = CommandRegistry()
    commands.register('new', 'Start over.', raising)
    gateway = FakeGateway()
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), gateway, commands=commands).process(
        channel(),
        (inbound(1, text='hello'), commanded(2, 'new')),
    )

    # NOTE: the owner is mid-drain, so a command must not raise out of the round; the
    # failed one wears the reaction and what was said still gets its answer
    assert gateway.abandoned == ['2']
    assert [prompt.text for prompt in turns.requests[0].prompts] == ['hello']


async def test_a_failed_round_reports_on_every_message_it_was_answering() -> None:
    gateway = FakeGateway()
    turns = FakeTurns(error=RuntimeError('llm stream died'))

    await make_processor(turns, FakeIdentities(), gateway).process(
        channel(),
        (inbound(1), inbound(2)),
    )

    # NOTE: the reply annotates the draft it already sent; the reaction tells each
    # message of the round that its answer is not coming
    assert gateway.replies[0].abandoned
    assert gateway.abandoned == ['1', '2']


async def test_a_failed_round_does_not_escape_the_processor() -> None:
    # NOTE: the owner is mid-drain, so a raised exception would cost the queue behind
    # this round the answers it is still owed
    processor = make_processor(
        FakeTurns(error=RuntimeError('llm stream died')),
        FakeIdentities(),
        FakeGateway(),
    )

    await processor.process(channel(), (inbound(1),))


async def test_an_empty_round_does_nothing() -> None:
    gateway = FakeGateway()
    turns = FakeTurns()

    await make_processor(turns, FakeIdentities(), gateway).process(channel(), ())

    assert turns.requests == []
    assert gateway.replies == []


async def test_a_failed_report_leaves_the_round_finished() -> None:
    # NOTE: telegram being unreachable is exactly when the turn failed in the first
    # place, so reporting the failure must not become a second unhandled one
    class RefusingGateway(FakeGateway):
        @override
        async def abandon(self, message: InboundMessage) -> None:
            raise RuntimeError('reaction refused')

    processor = make_processor(
        FakeTurns(error=RuntimeError('llm stream died')),
        FakeIdentities(),
        RefusingGateway(),
    )

    await processor.process(channel(), (inbound(1),))
