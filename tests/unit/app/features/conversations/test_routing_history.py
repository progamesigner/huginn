from uuid import UUID

from app.features.conversations.entities import Part, Turn
from app.features.conversations.services import routing_turn


def test_routing_history_excludes_tools_and_keeps_transcripts() -> None:
    turn = Turn(
        UUID(int=1),
        'coder',
        (
            Part('user', 'text', {'text': 'question'}),
            Part('user', 'audio', {'text': 'spoken question', 'blob': 'secret'}),
            Part('user', 'document', {'text': 'document hint'}),
            Part('assistant', 'text', {'text': 'final'}),
            Part('assistant', 'tool_call', {'text': 'private tool arguments'}),
            Part('tool', 'tool_result', {'text': 'private tool results'}),
            Part('assistant', 'summary', {'text': 'not a reply'}),
        ),
    )
    context = routing_turn(turn)
    assert context.user == ('question', 'spoken question')
    assert context.answer == 'final'
    assert context.attachments == ('audio', 'document')
