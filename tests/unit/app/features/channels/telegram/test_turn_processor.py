from typing import override
from uuid import UUID

from pydantic import SecretStr

from app.features.agents.contracts import TurnHandlerContract
from app.features.agents.types import TurnRequest
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.channels.telegram.turns import TelegramTurnProcessor
from app.features.identities.contracts import IdentityResolverContract
from app.features.identities.types import (
    AnonymousPrincipal,
    LinkedPrincipal,
    Principal,
)
from app.ports.channels import (
    ChannelGatewayContract,
    ChannelReplyContract,
    InboundMessage,
)

_LINKED = UUID('00000000-0000-7000-8000-000000000001')


class FakeReply(ChannelReplyContract):
    abandoned: bool
    finished: bool
    fragments: list[str]

    def __init__(self) -> None:
        self.abandoned = False
        self.finished = False
        self.fragments = []

    @override
    async def abandon(self) -> None:
        self.abandoned = True

    @override
    async def finish(self) -> None:
        self.finished = True

    @override
    async def on_text(self, fragment: str) -> None:
        self.fragments.append(fragment)


class FakeGateway(ChannelGatewayContract):
    abandoned: list[str]
    acknowledged: list[str]
    replies: list[FakeReply]

    def __init__(self) -> None:
        self.abandoned = []
        self.acknowledged = []
        self.replies = []

    @override
    async def abandon(self, message: InboundMessage) -> None:
        self.abandoned.append(message.id)

    @override
    async def acknowledge(self, message: InboundMessage) -> None:
        self.acknowledged.append(message.id)

    @override
    def reply(self, message: InboundMessage) -> ChannelReplyContract:
        reply = FakeReply()
        self.replies.append(reply)
        return reply


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


def inbound(
    event: int,
    *,
    text: str = 'hello',
    user: str | None = '77',
) -> InboundMessage:
    return InboundMessage(
        channel='support_bot',
        chat='4242',
        event=str(event),
        id=str(event),
        text=text,
        user=user,
    )


def make_processor(
    turns: FakeTurns,
    identities: FakeIdentities,
    gateway: FakeGateway,
) -> TelegramTurnProcessor:
    return TelegramTurnProcessor(
        turns=turns,
        identities=identities,
        gateway=lambda _: gateway,
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
