from collections.abc import Callable, Iterator

from app.features.identities.contracts import (
    IdentityLinkingContract,
    IdentityResolverContract,
)
from app.features.identities.exceptions import InvalidIdentityLinkError
from app.features.identities.types import AnonymousPrincipal, LinkedPrincipal
from app.ports.channels import InboundCommand, InboundMessage

from .registry import CommandHandler, CommandRegistry

type LinkerFactory = Callable[[], IdentityLinkingContract]
type ResolverFactory = Callable[[], IdentityResolverContract]

_LINK_PREFIX = 'link_'

_HELP = 'List the commands I understand.'
_IDENTITY = 'Show which account this chat is linked to.'
_REGISTER = 'Link this chat to your account with a one-time code.'
_START = 'Say hello and explain how to get started.'

_ANONYMOUS = (
    'This chat is not linked to an account yet.'
    ' Send /register with the code you were given.'
)
_MISSING_CODE = 'Send the code along with it, like /register ABCD-EFGH-1234.'
_UNIDENTIFIED = 'Telegram did not tell me who sent that, so I cannot link this chat.'
_WELCOME = (
    'Hello. Send me a message and I will answer.'
    ' Use /help to see what else I understand.'
)


class CommandHandlers:
    identities: ResolverFactory
    linking: LinkerFactory
    registry: CommandRegistry

    def __init__(
        self,
        identities: ResolverFactory,
        linking: LinkerFactory,
        registry: CommandRegistry,
    ) -> None:
        self.identities = identities
        self.linking = linking
        self.registry = registry

    def commands(self) -> Iterator[tuple[str, str, CommandHandler]]:
        yield 'help', _HELP, self.help
        yield 'identity', _IDENTITY, self.identity
        yield 'register', _REGISTER, self.register
        yield 'start', _START, self.start

    async def help(self, command: InboundCommand, message: InboundMessage) -> str:
        lines = (
            f'/{registered.name} — {registered.description}'
            for registered in self.registry.listing()
        )

        return '\n'.join(('Here is what I understand:', *lines))

    async def identity(self, command: InboundCommand, message: InboundMessage) -> str:
        if message.user is None:
            return _UNIDENTIFIED

        principal = await self.identities().resolve(
            message.kind,
            message.channel,
            message.user,
        )
        match principal:
            case LinkedPrincipal(user_id=user_id):
                return f'This chat is linked to account {user_id}.'
            case AnonymousPrincipal():
                return _ANONYMOUS

    async def register(self, command: InboundCommand, message: InboundMessage) -> str:
        if not command.args:
            return _MISSING_CODE

        return await self._link(command.args, message)

    async def start(self, command: InboundCommand, message: InboundMessage) -> str:
        if command.args.startswith(_LINK_PREFIX):
            return await self._link(command.args.removeprefix(_LINK_PREFIX), message)

        return _WELCOME

    async def _link(self, credential: str, message: InboundMessage) -> str:
        if message.user is None:
            return _UNIDENTIFIED

        try:
            principal = await self.linking().link(
                credential,
                message.kind,
                message.channel,
                message.user,
            )
        except InvalidIdentityLinkError as error:
            return str(error)

        return f'Done. This chat is linked to account {principal.user_id}.'
