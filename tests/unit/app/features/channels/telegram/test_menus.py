import asyncio
from typing import override
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.contracts.channels import (
    ChannelGatewayContract,
    ChannelGatewayFactoryContract,
    ChannelMenuContract,
    InboundMessage,
)
from app.features.agents.contracts import CommandInfo, ScopedCommandsServiceContract
from app.features.agents.entities import RoutingPolicy, TurnRequest
from app.features.channels.registry import Command, CommandRegistry
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.channels.telegram.menus import TelegramMenuService
from app.features.identities.contracts import IdentityResolutionServiceContract
from app.features.identities.entities import (
    AnonymousPrincipal,
    LinkedPrincipal,
    Principal,
)
from tests.helpers.logging import make_logger
from tests.unit.app.features.channels.telegram.test_turn_processor import FakeGateway


class Menus(FakeGateway, ChannelMenuContract):
    def __init__(self) -> None:
        super().__init__()
        self.calls = list[tuple[str, str, str | None, tuple[tuple[str, str], ...]]]()
        self.failures = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    @override
    async def set_commands(
        self,
        chat: str,
        member: str | None,
        commands: tuple[tuple[str, str], ...],
        language: str = '',
    ) -> None:
        self.started.set()
        await self.release.wait()
        if self.failures:
            self.failures -= 1
            raise RuntimeError('network failure')
        self.calls.append(('set', chat, member, commands))

    @override
    async def delete_commands(
        self,
        chat: str,
        member: str | None,
        language: str = '',
    ) -> None:
        self.calls.append(('delete', chat, member, ()))


class Factory(ChannelGatewayFactoryContract):
    def __init__(self, gateway: Menus) -> None:
        self.menu = gateway

    @override
    def gateway(self, token: SecretStr) -> ChannelGatewayContract:
        return self.menu


class Identities(IdentityResolutionServiceContract):
    def __init__(self) -> None:
        self.linked = True
        self.scope = 'existing-memory'

    @override
    async def resolve(
        self,
        provider: str,
        realm: str,
        external_user_id: str,
    ) -> Principal:
        if not self.linked:
            return AnonymousPrincipal(
                provider=provider,
                realm=realm,
                external_user_id=external_user_id,
            )
        return LinkedPrincipal(
            user_id=UUID(int=int(external_user_id)),
            external_identity_id=UUID(int=1),
            handle='new-handle',
            memory_scope=self.scope,
        )


class Commands(ScopedCommandsServiceContract):
    def __init__(self) -> None:
        self.description = 'Personal command'
        self.allowed = True
        self.enabled_for = True
        self.scopes = []

    @override
    def enabled(self, policy: RoutingPolicy) -> bool:
        return self.enabled_for

    @override
    async def inventory(self, request: TurnRequest) -> tuple[CommandInfo, ...]:
        prompt = request.prompts[0]
        self.scopes.append(prompt.memory_scope)
        return (
            (CommandInfo('personal', self.description),)
            if prompt.account and self.allowed
            else ()
        )

    @override
    async def resolve(self, request: TurnRequest) -> TurnRequest:
        return request


async def help_handler(*args: object) -> str:
    return 'help'


def channel(bot: str = 'bot') -> TelegramChannelConfig:
    return TelegramChannelConfig(
        kind='telegram',
        agent='writer',
        token=SecretStr(bot),
        secret_token=SecretStr('test'),
    )


def message(user: str = '1', private: bool = True) -> InboundMessage:
    return InboundMessage(
        channel='bot',
        chat=user if private else '-100',
        event='1',
        id='1',
        kind='telegram',
        user=user,
        text='hello',
        private=private,
    )


def service() -> tuple[TelegramMenuService, Menus, Identities, Commands]:
    menus, identities, commands = Menus(), Identities(), Commands()
    service = TelegramMenuService(
        commands,
        CommandRegistry((Command('help', 'Help', help_handler),)),
        identities,
        Factory(menus),
        logger=make_logger(),
    )
    service.backoff = 0
    return service, menus, identities, commands


async def test_dedup_personal_scope_refresh_and_unlink() -> None:
    sync, menus, identities, commands = service()
    await sync.synchronize(channel(), message())
    await sync.synchronize(channel(), message())
    assert len([call for call in menus.calls if call[0] == 'set']) == 1
    assert commands.scopes == ['existing-memory', 'existing-memory']
    commands.description = 'Edited description'
    await sync.synchronize(channel(), message())
    assert menus.calls[-1][3][-1] == ('personal', 'Edited description')
    identities.linked = False
    await sync.synchronize(channel(), message())
    assert menus.calls[-1][3] == (('help', 'Help'),)
    identities.linked = True
    commands.allowed = False
    await sync.synchronize(channel(), message())
    assert menus.calls[-1][3] == (('help', 'Help'),)


async def test_group_member_and_bot_do_not_share_menu_cache() -> None:
    sync, menus, _, _ = service()
    await sync.synchronize(channel(), message('1', False))
    await sync.synchronize(channel(), message('2', False))
    await sync.synchronize(channel('second-bot'), message('1', False))
    writes = [call for call in menus.calls if call[0] == 'set']
    assert [(call[1], call[2]) for call in writes] == [
        ('-100', '1'),
        ('-100', '2'),
        ('-100', '1'),
    ]


async def test_bounded_retries_clear_obsolete_entries_and_recover_next_interaction() -> (
    None
):
    sync, menus, _, _ = service()
    menus.failures = 3
    await sync.synchronize(channel(), message())
    assert not sync.fingerprints
    assert menus.calls[-1][0] == 'delete'
    await sync.synchronize(channel(), message())
    assert menus.calls[-1][0] == 'set'
    assert len(sync.fingerprints) == 1


async def test_queued_calculation_resolves_identity_after_the_previous_publish() -> (
    None
):
    sync, menus, identities, _ = service()
    menus.release.clear()
    first = asyncio.create_task(sync.synchronize(channel(), message()))
    await menus.started.wait()
    second = asyncio.create_task(sync.synchronize(channel(), message()))
    identities.linked = False
    menus.release.set()
    await asyncio.gather(first, second)
    assert menus.calls[-1][3] == (('help', 'Help'),)


async def test_failed_retries_remove_obsolete_entries_once_per_attempt() -> None:
    sync, menus, _, _ = service()
    menus.failures = 3
    await sync.synchronize(channel(), message())
    assert [call[0] for call in menus.calls] == ['delete', 'delete', 'delete']


async def test_a_stalled_chat_does_not_block_another_chat() -> None:
    stalled = asyncio.Event()

    class Stalling(Identities):
        @override
        async def resolve(
            self,
            provider: str,
            realm: str,
            external_user_id: str,
        ) -> Principal:
            if external_user_id == '1':
                await stalled.wait()
            return await super().resolve(provider, realm, external_user_id)

    sync, menus, _, _ = service()
    sync.identities = Stalling()
    first = asyncio.create_task(sync.synchronize(channel(), message('1')))
    await asyncio.sleep(0)
    async with asyncio.timeout(1):
        await sync.synchronize(channel(), message('2'))
    assert [call[1] for call in menus.calls if call[0] == 'set'] == ['2']
    stalled.set()
    await first
    assert not sync.locks


async def test_scheduling_runs_in_background_and_coalesces_queued_requests() -> None:
    sync, menus, _, _ = service()
    menus.release.clear()
    sync.schedule(channel(), message())
    await menus.started.wait()
    for _ in range(3):
        sync.schedule(channel(), message())
    assert len(sync.tasks.tasks) == 2
    menus.release.set()
    await sync.drain()
    assert menus.calls[-1][0] == 'set'
    assert not sync.locks
    assert not sync.pending


async def test_queued_run_gets_its_own_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.channels.telegram import menus as module

    monkeypatch.setattr(module, '_SYNC_TIMEOUT', 0.2)
    sync, menus, identities, _ = service()
    menus.release.clear()
    sync.schedule(channel(), message())
    await menus.started.wait()
    identities.linked = False
    sync.schedule(channel(), message())
    # Past the first run's limit, and past the limit had it counted queueing.
    await asyncio.sleep(0.3)
    menus.release.set()
    await sync.drain()
    # The first run timed out while stuck; the queued, newer state is published.
    assert menus.calls[-1] == ('set', '1', None, (('help', 'Help'),))


async def test_disabling_commands_withdraws_a_published_menu_once() -> None:
    sync, menus, _, commands = service()
    sync.schedule(channel(), message())
    await sync.drain()
    commands.enabled_for = False
    sync.schedule(channel(), message())
    sync.schedule(channel(), message('2'))
    await sync.drain()
    assert menus.calls[-1] == ('delete', '1', None, ())
    sync.schedule(channel(), message())
    await sync.dispose()
    assert [call[0] for call in menus.calls] == ['set', 'delete']


async def test_a_failed_lookup_keeps_the_published_menu() -> None:
    class Flaky(Commands):
        failing = False

        @override
        async def inventory(self, request: TurnRequest) -> tuple[CommandInfo, ...]:
            if self.failing:
                raise RuntimeError('memory service down')
            return await super().inventory(request)

    sync, menus, _, _ = service()
    flaky = Flaky()
    sync.commands = flaky
    await sync.synchronize(channel(), message())
    flaky.failing = True
    await sync.synchronize(channel(), message())
    assert [call[0] for call in menus.calls] == ['set']
    assert not sync.fingerprints
