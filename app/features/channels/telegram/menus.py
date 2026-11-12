import asyncio
import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

from app.contracts.channels import (
    ChannelGatewayFactoryContract,
    ChannelMenuContract,
    InboundMessage,
)
from app.features.agents.contracts import ScopedCommandsServiceContract
from app.features.agents.entities import TurnPrompt, TurnRequest
from app.features.channels.config import routing_policy
from app.features.channels.registry import CommandRegistry
from app.features.identities.contracts import IdentityResolutionServiceContract
from app.features.identities.entities import LinkedPrincipal
from lib.concurrency import TaskSupervisor
from lib.observability import Logger

from .config import TelegramChannelConfig

_NAME = re.compile(r'^[a-z0-9_]{1,32}$')
_SYNC_TIMEOUT = 30.0
_DRAIN_GRACE = timedelta(seconds=5)

type MenuKey = tuple[str, str, str | None]


class TelegramMenuService:
    def __init__(
        self,
        commands: ScopedCommandsServiceContract,
        builtins: CommandRegistry,
        identities: IdentityResolutionServiceContract,
        gateways: ChannelGatewayFactoryContract,
        *,
        logger: Logger,
    ) -> None:
        self.commands = commands
        self.builtins = builtins
        self.identities = identities
        self.gateways = gateways
        self.logger = logger.bind(component='telegram-menu')
        self.locks: dict[MenuKey, tuple[asyncio.Lock, int]] = {}
        self.fingerprints: OrderedDict[MenuKey, str] = OrderedDict()
        self.tasks = TaskSupervisor(logger=self.logger, grace=_DRAIN_GRACE)
        # Keys with a synchronization still waiting for its turn; that queued
        # run resolves identity afresh, so another request adds nothing.
        self.pending: set[MenuKey] = set()
        self.backoff = 0.5

    def schedule(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
    ) -> None:
        # A menu is a convenience: it never delays or fails the conversation.
        target = self._target(channel, message)
        if target is None or target[1] in self.pending:
            return
        gateway, key = target
        if not self.commands.enabled(routing_policy(channel)):
            # Withdraw a personal menu this process published before commands
            # were disabled; menus published before a restart are not tracked.
            if self.fingerprints.pop(key, None) is not None:
                self.pending.add(key)
                self.tasks.spawn(self._withdraw(gateway, key))
            return
        self.pending.add(key)
        self.tasks.spawn(self._bounded(channel, message, gateway, key))

    async def dispose(self) -> None:
        await self.tasks.dispose()

    async def drain(self) -> None:
        await self.tasks.drain()

    def _target(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
    ) -> tuple[ChannelMenuContract, MenuKey] | None:
        if message.user is None:
            return None
        gateway = self.gateways.gateway(channel.token)
        if not isinstance(gateway, ChannelMenuContract):
            return None
        member = None if message.private else message.user
        bot = hashlib.sha256(channel.token.get_secret_value().encode()).hexdigest()
        return gateway, (bot, message.chat, member)

    async def _withdraw(self, gateway: ChannelMenuContract, key: MenuKey) -> None:
        async with self._serial(key):
            self.pending.discard(key)
            try:
                async with asyncio.timeout(_SYNC_TIMEOUT):
                    await gateway.delete_commands(key[1], key[2])
            except Exception:
                self.logger.warning(
                    'Could not withdraw the personal menu',
                    exc_info=True,
                )

    async def _bounded(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
        gateway: ChannelMenuContract,
        key: MenuKey,
    ) -> None:
        async with self._serial(key):
            self.pending.discard(key)
            # The limit starts once this run holds the menu, not while queued.
            try:
                async with asyncio.timeout(_SYNC_TIMEOUT):
                    await self._publish(channel, message, gateway, key)
            except TimeoutError:
                self.fingerprints.pop(key, None)
                self.logger.warning(
                    'Menu synchronization timed out channel=%s',
                    message.channel,
                )
            except Exception:
                self.logger.exception(
                    'Menu synchronization failed channel=%s',
                    message.channel,
                )

    @asynccontextmanager
    async def _serial(self, key: MenuKey) -> AsyncIterator[None]:
        lock, users = self.locks.get(key, (asyncio.Lock(), 0))
        self.locks[key] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            lock, users = self.locks[key]
            if users == 1:
                del self.locks[key]
            else:
                self.locks[key] = (lock, users - 1)

    async def synchronize(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
    ) -> None:
        target = self._target(channel, message)
        if target is None:
            return
        gateway, key = target
        async with self._serial(key):
            await self._publish(channel, message, gateway, key)

    async def _publish(
        self,
        channel: TelegramChannelConfig,
        message: InboundMessage,
        gateway: ChannelMenuContract,
        key: MenuKey,
    ) -> None:
        # Runs inside the per-menu serial section and resolves identity there:
        # a queued old calculation never overwrites the menu computed after a
        # link or an unlink.
        user = message.user
        if user is None:
            return
        member = key[2]
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(self.backoff * 2 ** (attempt - 1))
            publishing = False
            try:
                principal = await self.identities.resolve(
                    message.kind,
                    message.channel,
                    user,
                )
                linked = principal if isinstance(principal, LinkedPrincipal) else None
                prompt = TurnPrompt(
                    text='',
                    actor=principal.stable_key,
                    account=str(linked.user_id) if linked else None,
                    handle=linked.handle if linked else None,
                    memory_scope=linked.memory_scope if linked else None,
                )
                request = TurnRequest(
                    channel=message.channel,
                    chat=message.chat,
                    policy=routing_policy(channel),
                    prompts=(prompt,),
                )
                scoped = await self.commands.inventory(request)
                listing = [
                    (item.name, item.description) for item in self.builtins.listing()
                ] + [(item.name, item.description) for item in scoped]
                publishable = []
                for name, description in listing:
                    if not _NAME.fullmatch(name) or not 1 <= len(description) <= 256:
                        self.logger.warning(
                            'Command /%s cannot appear in the Telegram menu; use /help',
                            name,
                        )
                        continue
                    publishable.append((name, description))
                if len(publishable) > 100:
                    self.logger.warning(
                        'Telegram menu capacity exceeded; use /help for the complete list',
                    )
                menu = tuple(publishable[:100])
                fingerprint = hashlib.sha256(
                    json.dumps(
                        (prompt.account, prompt.memory_scope, menu),
                    ).encode(),
                ).hexdigest()
                if self.fingerprints.get(key) == fingerprint:
                    self.fingerprints.move_to_end(key)
                    return
                # Setting replaces the scope's menu; a failure falls through to
                # removing obsolete entries. A menu is never an authorization cache.
                publishing = True
                await gateway.set_commands(message.chat, member, menu)
                self.fingerprints[key] = fingerprint
                self.fingerprints.move_to_end(key)
                while len(self.fingerprints) > 256:
                    self.fingerprints.popitem(last=False)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                self.fingerprints.pop(key, None)
                self.logger.warning(
                    'Menu synchronization failed attempt=%d channel=%s',
                    attempt + 1,
                    message.channel,
                    exc_info=True,
                )
                # Only a failed replacement leaves the menu in an unknown state;
                # a failed lookup keeps the published menu until the next try.
                if not publishing:
                    continue
                try:
                    await gateway.delete_commands(message.chat, member)
                except Exception:
                    self.logger.warning(
                        'Could not remove the obsolete menu channel=%s',
                        message.channel,
                        exc_info=True,
                    )
