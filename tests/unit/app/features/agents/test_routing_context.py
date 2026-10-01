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
        turns=(RoutingTurn(agent='coder', user=('old',), answer='x' * 900),),
        summary='summary',
    )
    state = routing_state(context)
    assert state['current'] == ['first', 'second']
    assert state['summary'] == 'summary'
    assert 'coder' in json.dumps(state)
    assert 'x' * 601 not in json.dumps(state)


def test_budget_drops_oldest_then_truncates_current() -> None:
    turns = (
        RoutingTurn(agent='old', user=('x' * 600,), answer=''),
        RoutingTurn(agent='new', user=('hi',), answer=''),
    )
    state = routing_state(
        RoutingContext(current=(TurnPrompt(text='now', actor='u'),), turns=turns),
        400,
    )
    assert 'old' not in json.dumps(state)
    assert 'new' in json.dumps(state)
    state = routing_state(
        RoutingContext(current=(TurnPrompt(text='"' * 10000, actor='u'),), turns=turns),
        400,
    )
    assert len(json.dumps(state, ensure_ascii=False, separators=(',', ':'))) <= 400
    assert '…' in json.dumps(state, ensure_ascii=False)


def test_attachment_state_contains_hints_and_transcripts_only() -> None:
    attachment = PromptAttachment(
        kind='document',
        mime='text/plain',
        reference='private-reference',
        unique='unique',
        data=b'private-bytes',
        hint='document',
        name='notes.txt',
        document_text='private-body',
    )
    context = RoutingContext(
        current=(TurnPrompt(text='read this', actor='u', attachments=(attachment,)),),
    )
    serialized = json.dumps(routing_state(context))
    assert 'notes.txt' in serialized
    assert 'private' not in serialized
