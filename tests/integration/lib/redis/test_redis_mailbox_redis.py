import asyncio
import secrets
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from lib.redis import RedisLock, RedisMailbox, RedisManager
from tests.helpers.logging import make_logger

pytestmark = pytest.mark.integration

_TTL = timedelta(milliseconds=300)


@pytest.fixture
async def redis(redis_url: str) -> AsyncIterator[RedisManager]:
    manager = RedisManager({None: redis_url})
    try:
        yield manager
    finally:
        await manager.dispose()


@pytest.fixture
def chat() -> str:
    return f'huginn:test:mailbox:{secrets.token_hex(8)}'


def make_mailbox(redis: RedisManager, chat: str) -> RedisMailbox:
    return RedisMailbox(
        RedisLock(redis, f'{chat}:lock', ttl=_TTL, logger=make_logger()),
        f'{chat}:queue',
    )


async def backlog(redis: RedisManager, chat: str) -> int:
    async with redis.connect() as client:
        return await client.llen(f'{chat}:queue')


async def locked(redis: RedisManager, chat: str) -> bool:
    async with redis.connect() as client:
        return bool(await client.exists(f'{chat}:lock'))


async def assert_invariant(redis: RedisManager, chat: str) -> None:
    if await backlog(redis, chat) > 0:
        assert await locked(redis, chat), 'a queued message has nobody to process it'


async def test_the_first_claim_owns_the_turn_and_the_rest_queue(
    redis: RedisManager,
    chat: str,
) -> None:
    owner = make_mailbox(redis, chat)
    caller = make_mailbox(redis, chat)

    assert await owner.claim(b'first') is True
    assert await caller.claim(b'second') is False

    assert await owner.drain() == (b'second',)
    await assert_invariant(redis, chat)

    assert await owner.drain() == ()
    assert await locked(redis, chat) is False


async def test_drain_coalesces_the_whole_backlog_into_one_batch(
    redis: RedisManager,
    chat: str,
) -> None:
    owner = make_mailbox(redis, chat)

    assert await owner.claim(b'first') is True
    for message in (b'second', b'third', b'fourth'):
        assert await make_mailbox(redis, chat).claim(message) is False

    # NOTE: one turn per drain — many bubbles collapse into a single thought
    assert await owner.drain() == (b'second', b'third', b'fourth')
    assert await owner.drain() == ()


async def test_a_lapsed_owner_neither_releases_nor_robs_its_successor(
    redis: RedisManager,
    chat: str,
) -> None:
    lapsed = make_mailbox(redis, chat)
    successor = make_mailbox(redis, chat)

    assert await lapsed.claim(b'first') is True
    await asyncio.sleep(_TTL.total_seconds() * 2)
    assert await successor.claim(b'second') is True
    assert await make_mailbox(redis, chat).claim(b'third') is False

    assert await lapsed.drain() == ()
    assert await locked(redis, chat) is True

    assert await successor.drain() == (b'third',)
    await assert_invariant(redis, chat)


async def test_an_orphaned_backlog_is_picked_up_by_the_next_owner(
    redis: RedisManager,
    chat: str,
) -> None:
    crashed = make_mailbox(redis, chat)
    successor = make_mailbox(redis, chat)

    assert await crashed.claim(b'first') is True
    assert await make_mailbox(redis, chat).claim(b'second') is False
    await asyncio.sleep(_TTL.total_seconds() * 2)

    # NOTE: crash recovery — the lease lapses, the next message adopts the backlog
    assert await successor.claim(b'third') is True
    assert await successor.drain() == (b'second',)
    await assert_invariant(redis, chat)


async def test_a_message_arriving_before_the_drain_survives_the_release(
    redis: RedisManager,
    chat: str,
) -> None:
    owner = make_mailbox(redis, chat)
    caller = make_mailbox(redis, chat)

    assert await owner.claim(b'first') is True
    assert await caller.claim(b'second') is False

    # NOTE: the check and the release live in one script, so no interleaving can split them
    assert await owner.drain() == (b'second',)
    assert await locked(redis, chat) is True
    await assert_invariant(redis, chat)


async def test_a_split_drain_would_strand_the_message_it_never_saw(
    redis: RedisManager,
    chat: str,
) -> None:
    owner = make_mailbox(redis, chat)
    caller = make_mailbox(redis, chat)

    assert await owner.claim(b'first') is True

    # NOTE: the control group — LLEN and DEL as two round trips is the lost wakeup
    async with redis.connect() as client:
        assert await client.llen(f'{chat}:queue') == 0
        assert await caller.claim(b'second') is False
        await client.delete(f'{chat}:lock')

    assert await backlog(redis, chat) == 1
    assert await locked(redis, chat) is False


async def test_a_split_claim_would_queue_behind_an_owner_that_already_left(
    redis: RedisManager,
    chat: str,
) -> None:
    owner = make_mailbox(redis, chat)
    caller = make_mailbox(redis, chat)

    assert await owner.claim(b'first') is True

    # NOTE: the other half of the race — acquire then push strands the message too
    async with redis.connect() as client:
        assert await caller.lock.acquire() is False
        assert await owner.drain() == ()
        await client.rpush(f'{chat}:queue', b'second')

    assert await backlog(redis, chat) == 1
    assert await locked(redis, chat) is False

    # NOTE: claim.lua closes it — the same ordering hands the caller the turn instead
    assert await caller.claim(b'third') is True
    assert await caller.drain() == (b'second',)
