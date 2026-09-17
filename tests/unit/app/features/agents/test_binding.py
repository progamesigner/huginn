from app.features.agents.config import BindingSource
from app.features.agents.types import TurnScope, resolve_bindings


def scope(**changes: str | None) -> TurnScope:
    values = {
        'actor': 'user:1',
        'agent': 'assistant',
        'channel': 'support',
        'chat': '42',
        'account': '00000000-0000-0000-0000-000000000001',
        'handle': 'alice',
    }
    values.update(changes)
    return TurnScope(
        actor=str(values['actor'] or ''),
        agent=str(values['agent'] or ''),
        channel=str(values['channel'] or ''),
        chat=str(values['chat'] or ''),
        account=values['account'],
        handle=values['handle'],
    )


def binding(source: BindingSource) -> dict[str, BindingSource]:
    return {'user': source}


def all_bindings() -> dict[str, BindingSource]:
    return {
        'account_arg': 'account',
        'actor_arg': 'actor',
        'agent_arg': 'agent',
        'channel_arg': 'channel',
        'chat_arg': 'chat',
        'handle_arg': 'handle',
    }


def test_plain_injection_is_unchanged() -> None:
    assert resolve_bindings({'agent': 'default'}, {}, scope()) == {
        'agent': 'default',
    }


def test_every_turn_binding_source_resolves() -> None:
    resolved = resolve_bindings(
        {},
        all_bindings(),
        scope(),
    )

    assert resolved == {
        'account_arg': '00000000-0000-0000-0000-000000000001',
        'actor_arg': 'user:1',
        'agent_arg': 'assistant',
        'channel_arg': 'support',
        'chat_arg': '42',
        'handle_arg': 'alice',
    }


def test_missing_optional_identity_withholds_the_binding() -> None:
    assert resolve_bindings({}, binding('handle'), scope(handle=None)) is None
    assert resolve_bindings({}, binding('account'), scope(account=None)) is None


def test_empty_actor_withholds_the_binding() -> None:
    assert resolve_bindings({}, binding('actor'), scope(actor='')) is None


def test_unsafe_path_like_value_withholds_the_binding() -> None:
    assert (
        resolve_bindings(
            {},
            binding('handle'),
            scope(handle='../../alice'),
        )
        is None
    )
