from typing import override
from uuid import UUID

from app.contracts.channels import InboundCommand, InboundMessage
from app.features.channels.handlers import CommandHandlers
from app.features.channels.registry import CommandRegistry
from app.features.conversations.contracts import ConversationControlContract
from app.features.identities.contracts import (
    IdentityLinkingContract,
    IdentityResolverContract,
)
from app.features.identities.exceptions import InvalidIdentityLinkError
from app.features.identities.types import (
    AnonymousPrincipal,
    LinkedPrincipal,
    Principal,
)

_LINKED = UUID('00000000-0000-7000-8000-000000000001')
_IDENTITY = UUID('00000000-0000-7000-8000-000000000002')


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
            return LinkedPrincipal(_LINKED, _IDENTITY, 'alice')

        return AnonymousPrincipal(provider, realm, external_user_id)


class FakeControl(ConversationControlContract):
    cleared: bool
    compactions: list[tuple[str, str]]
    folded: bool
    resets: list[tuple[str, str]]

    def __init__(self, *, cleared: bool = True, folded: bool = True) -> None:
        self.cleared = cleared
        self.compactions = []
        self.folded = folded
        self.resets = []

    @override
    async def compact(self, channel: str, chat: str) -> bool:
        self.compactions.append((channel, chat))

        return self.folded

    @override
    async def reset(self, channel: str, chat: str) -> bool:
        self.resets.append((channel, chat))

        return self.cleared


class FakeLinking(IdentityLinkingContract):
    calls: list[tuple[str, str, str, str]]
    invalid: bool

    def __init__(self, *, invalid: bool = False) -> None:
        self.calls = []
        self.invalid = invalid

    @override
    async def link(
        self,
        credential: str,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> LinkedPrincipal:
        self.calls.append((credential, provider, realm, external_user_id))
        if self.invalid:
            raise InvalidIdentityLinkError('unknown_token')

        return LinkedPrincipal(_LINKED, _IDENTITY, 'alice')


def command(name: str, args: str = '') -> InboundCommand:
    return InboundCommand(name=name, args=args)


def inbound(user: str | None = '77') -> InboundMessage:
    return InboundMessage(
        channel='support_bot',
        chat='4242',
        event='7',
        id='7',
        kind='telegram',
        text='/start',
        user=user,
    )


def make_handlers(
    identities: FakeIdentities | None = None,
    linking: FakeLinking | None = None,
    control: FakeControl | None = None,
) -> tuple[CommandHandlers, CommandRegistry]:
    registry = CommandRegistry()
    handlers = CommandHandlers(
        lambda: identities or FakeIdentities(),
        lambda: linking or FakeLinking(),
        lambda: control or FakeControl(),
        registry,
    )
    for name, description, handler in handlers.commands():
        registry.register(name, description, handler)

    return handlers, registry


async def test_help_lists_every_registered_command_with_its_description() -> None:
    handlers, _ = make_handlers()

    answer = await handlers.help(command('help'), inbound())

    # NOTE: the description is the user-facing documentation of the command, the same
    # text the telegram menu will carry, so /help has to read the registry rather than
    # a second list that drifts from it
    assert answer.splitlines() == [
        'Here is what I understand:',
        '/compact — Summarise this chat now to free up context.',
        '/help — List the commands I understand.',
        '/identity — Show which account this chat is linked to.',
        '/new — Start over: forget what we have said so far.',
        '/register — Link this chat to your account with a one-time code.',
        '/start — Say hello and explain how to get started.',
    ]


async def test_start_greets_when_it_carries_no_deep_link() -> None:
    handlers, _ = make_handlers()

    answer = await handlers.start(command('start'), inbound())

    assert 'Hello' in answer
    assert '/help' in answer


async def test_start_completes_the_deep_link_it_was_opened_with() -> None:
    linking = FakeLinking()
    handlers, _ = make_handlers(linking=linking)

    answer = await handlers.start(command('start', 'link_TOKEN123'), inbound())

    # NOTE: telegram delivers t.me/bot?start=link_<token> as '/start link_<token>', so
    # the prefix is telegram's envelope and the credential is what is left
    assert linking.calls == [('TOKEN123', 'telegram', 'support_bot', '77')]
    assert str(_LINKED) in answer


async def test_register_links_with_the_code_it_was_given() -> None:
    linking = FakeLinking()
    handlers, _ = make_handlers(linking=linking)

    answer = await handlers.register(command('register', 'ABCD-EFGH-1234'), inbound())

    assert linking.calls == [('ABCD-EFGH-1234', 'telegram', 'support_bot', '77')]
    assert str(_LINKED) in answer


async def test_register_without_a_code_says_how_to_use_it() -> None:
    linking = FakeLinking()
    handlers, _ = make_handlers(linking=linking)

    answer = await handlers.register(command('register'), inbound())

    assert linking.calls == []
    assert '/register' in answer


async def test_a_refused_code_answers_with_what_the_service_said() -> None:
    handlers, _ = make_handlers(linking=FakeLinking(invalid=True))

    answer = await handlers.register(command('register', 'STALE'), inbound())

    # NOTE: the service already words this for a person; rewording it here would mean
    # two sentences to keep in step
    assert answer == str(InvalidIdentityLinkError('unknown_token'))


async def test_identity_reports_the_account_of_a_linked_chat() -> None:
    identities = FakeIdentities(linked=frozenset({'77'}))
    handlers, _ = make_handlers(identities=identities)

    answer = await handlers.identity(command('identity'), inbound())

    assert identities.calls == [('telegram', 'support_bot', '77')]
    assert str(_LINKED) in answer


async def test_identity_tells_an_unlinked_chat_how_to_link() -> None:
    handlers, _ = make_handlers(identities=FakeIdentities())

    answer = await handlers.identity(command('identity'), inbound())

    assert '/register' in answer


async def test_a_sender_telegram_did_not_name_cannot_be_linked() -> None:
    linking = FakeLinking()
    handlers, _ = make_handlers(linking=linking)

    answer = await handlers.register(command('register', 'ABC'), inbound(user=None))

    assert linking.calls == []
    assert 'cannot link' in answer


async def test_the_provider_of_a_link_is_the_vendor_the_message_named() -> None:
    linking = FakeLinking()
    handlers, _ = make_handlers(linking=linking)
    message = inbound().model_copy(update={'kind': 'http'})

    await handlers.register(command('register', 'ABC'), message)

    # NOTE: the realm pair is (kind, channel) and a second transport is coming, so the
    # handler must read the vendor off the message rather than assume telegram
    assert linking.calls == [('ABC', 'http', 'support_bot', '77')]


async def test_new_resets_every_conversation_of_the_chat() -> None:
    control = FakeControl(cleared=True)
    handlers, _ = make_handlers(control=control)

    answer = await handlers.new(command('new'), inbound())

    # NOTE: a chat can hold one conversation per agent, and the user means this chat --
    # so the command names the chat and the count comes back from conversations
    assert control.resets == [('support_bot', '4242')]
    assert 'forgotten' in answer


async def test_new_says_so_when_there_was_nothing_to_forget() -> None:
    handlers, _ = make_handlers(control=FakeControl(cleared=False))

    answer = await handlers.new(command('new'), inbound())

    assert 'nothing' in answer


async def test_compact_folds_every_conversation_of_the_chat() -> None:
    control = FakeControl(folded=True)
    handlers, _ = make_handlers(control=control)

    answer = await handlers.compact(command('compact'), inbound())

    assert control.compactions == [('support_bot', '4242')]
    assert 'Summarised' in answer


async def test_compact_says_so_when_there_was_nothing_to_fold() -> None:
    handlers, _ = make_handlers(control=FakeControl(folded=False))

    answer = await handlers.compact(command('compact'), inbound())

    assert 'nothing' in answer
