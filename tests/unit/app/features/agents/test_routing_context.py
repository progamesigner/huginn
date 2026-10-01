import json

from app.features.agents.entities import (
    PromptAttachment,
    RoutingContext,
    RoutingTurn,
    TurnPrompt,
)
from app.features.agents.routing_context import render_routing_state, routing_state


def test_legacy_message_is_unchanged() -> None:
    context = RoutingContext(current=(TurnPrompt(text='hello\n世界', actor='u'),))
    assert render_routing_state(context) == 'hello\n世界'


def test_multiple_messages_and_history() -> None:
    context = RoutingContext(
        current=(
            TurnPrompt(text='first', actor='u'),
            TurnPrompt(text='second', actor='u'),
        ),
        turns=(RoutingTurn('coder', ('old',), 'x' * 900),),
        summary='summary',
    )
    state = routing_state(context)
    assert state['current'] == ['first', 'second']
    assert state['summary'] == 'summary'
    assert 'coder' in json.dumps(state)
    assert 'x' * 601 not in json.dumps(state)


def test_budget_drops_oldest_then_truncates_current() -> None:
    turns = (RoutingTurn('old', ('x' * 600,), ''), RoutingTurn('new', ('hi',), ''))
    state = routing_state(RoutingContext((TurnPrompt('now', 'u'),), turns), 400)
    assert 'old' not in json.dumps(state)
    assert 'new' in json.dumps(state)
    state = routing_state(RoutingContext((TurnPrompt('"' * 10000, 'u'),), turns), 400)
    assert len(json.dumps(state, ensure_ascii=False, separators=(',', ':'))) <= 400
    assert '…' in json.dumps(state, ensure_ascii=False)


def test_attachment_state_contains_hints_and_transcripts_only() -> None:
    attachment = PromptAttachment(
        'document',
        'text/plain',
        'private-reference',
        'unique',
        b'private-bytes',
        'document',
        name='notes.txt',
        document_text='private-body',
    )
    context = RoutingContext((TurnPrompt('read this', 'u', (attachment,)),))
    serialized = json.dumps(routing_state(context))
    assert 'notes.txt' in serialized
    assert 'private' not in serialized
