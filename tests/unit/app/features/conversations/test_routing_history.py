from uuid import UUID

from app.features.conversations.entities import Part, Turn
from app.features.conversations.services import routing_turn


def test_routing_history_excludes_tools_and_keeps_transcripts() -> None:
    turn = Turn(
        id=UUID(int=1),
        agent='coder',
        parts=(
            Part(role='user', kind='text', payload={'text': 'question'}),
            Part(
                role='user',
                kind='audio',
                payload={'text': 'spoken question', 'blob': 'secret'},
            ),
            Part(role='user', kind='document', payload={'text': 'document hint'}),
            Part(role='assistant', kind='text', payload={'text': 'final'}),
            Part(
                role='assistant',
                kind='tool_call',
                payload={'text': 'private tool arguments'},
            ),
            Part(
                role='tool',
                kind='tool_result',
                payload={'text': 'private tool results'},
            ),
            Part(role='assistant', kind='summary', payload={'text': 'not a reply'}),
        ),
    )
    context = routing_turn(turn)
    assert context.user == ('question', 'spoken question')
    assert context.answer == 'final'
    assert context.attachments == ('audio', 'document')
