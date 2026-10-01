from datetime import timedelta

from lib.redis import RedisLock, RedisMailbox
from lib.redis.scripts import CLAIM, DRAIN
from tests.helpers.logging import make_logger
from tests.helpers.redis import FakeClient, make_redis

_KEYS = ('chat:7:lock', 'chat:7:queue')
_MESSAGE = b'{"update_id": 41}'
_TTL = timedelta(milliseconds=60)


def make_mailbox(client: FakeClient) -> RedisMailbox:
    return RedisMailbox(
        RedisLock(make_redis(client), _KEYS[0], ttl=_TTL, logger=make_logger()),
        _KEYS[1],
    )


def test_claim_takes_the_lock_and_enqueues_in_a_single_script() -> None:
    assert "redis.call('set', KEYS[1], ARGV[1], 'nx', 'px', ARGV[2])" in CLAIM
    assert "redis.call('rpush', KEYS[2], ARGV[3])" in CLAIM


def test_drain_guards_on_the_owner_token_before_touching_either_key() -> None:
    assert "redis.call('get', KEYS[1]) ~= ARGV[1]" in DRAIN

    guard = DRAIN.index("redis.call('get', KEYS[1])")
    for command in ("redis.call('llen', KEYS[2])", "redis.call('del', KEYS[1])"):
        assert DRAIN.index(command) > guard


def test_drain_releases_the_lock_and_pops_the_backlog_in_a_single_script() -> None:
    assert "redis.call('llen', KEYS[2]) == 0" in DRAIN
    assert "redis.call('del', KEYS[1])" in DRAIN
    assert "redis.call('lrange', KEYS[2], 0, -1)" in DRAIN
    assert "redis.call('del', KEYS[2])" in DRAIN


async def test_a_claimed_turn_belongs_to_the_caller() -> None:
    client = FakeClient(outcomes=[1])
    mailbox = make_mailbox(client)

    assert await mailbox.claim(_MESSAGE) is True
    assert client.evaluated == [
        (CLAIM, _KEYS, (mailbox.lock.token, 60, _MESSAGE)),
    ]


async def test_a_refused_claim_leaves_the_message_queued() -> None:
    client = FakeClient(outcomes=[0])
    mailbox = make_mailbox(client)

    assert await mailbox.claim(_MESSAGE) is False


async def test_drain_hands_back_the_coalesced_backlog() -> None:
    client = FakeClient(outcomes=[[b'first', b'second']])
    mailbox = make_mailbox(client)

    assert await mailbox.drain() == (b'first', b'second')
    assert client.evaluated == [(DRAIN, _KEYS, (mailbox.lock.token,))]


async def test_an_empty_drain_means_the_lock_is_no_longer_held() -> None:
    client = FakeClient(outcomes=[[]])
    mailbox = make_mailbox(client)

    assert await mailbox.drain() == ()


async def test_both_scripts_carry_the_same_owner_token() -> None:
    client = FakeClient(outcomes=[1, []])
    mailbox = make_mailbox(client)

    await mailbox.claim(_MESSAGE)
    await mailbox.drain()

    claimed, drained = client.evaluated
    assert claimed[2][0] == drained[2][0] == mailbox.lock.token


async def test_the_mailbox_follows_the_lock_to_its_alias() -> None:
    redis = make_redis(FakeClient(outcomes=[1]))
    lock = RedisLock(redis, _KEYS[0], ttl=_TTL, alias='queue', logger=make_logger())

    await RedisMailbox(lock, _KEYS[1]).claim(_MESSAGE)

    redis.connect.assert_called_once_with('queue')
