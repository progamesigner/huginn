from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast
from uuid import UUID

from pydantic import JsonValue

from app.features.conversations.exceptions import NoTurnsToCompactError
from app.features.conversations.types import Conversation, Part, Turn


class FakeConversationRepository:
    conversations: dict[tuple[str, str, str], Conversation]
    turns: dict[UUID, list[Turn]]
    counter: int
    force_cas_conflict: bool

    def __init__(self) -> None:
        self.conversations = {}
        self.turns = {}
        self.counter = 0
        self.force_cas_conflict = False

    def _next_id(self) -> UUID:
        self.counter += 1
        return UUID(int=self.counter)

    async def get_or_create(
        self,
        channel: str,
        chat_id: str,
        agent: str,
    ) -> Conversation:
        key = (channel, chat_id, agent)
        if key not in self.conversations:
            conversation = Conversation(
                id=self._next_id(),
                channel=channel,
                chat_id=chat_id,
                agent=agent,
            )
            self.conversations[key] = conversation
            self.turns[conversation.id] = []

        return self.conversations[key]

    async def append_turn(
        self,
        conversation: Conversation,
        parts: Sequence[Part],
        usage: Mapping[str, JsonValue] | None = None,
    ) -> Turn:
        turn = Turn(id=self._next_id(), parts=tuple(parts), usage=dict(usage or {}))
        self.turns[conversation.id].append(turn)
        return turn

    async def load_recent_turns(
        self,
        conversation: Conversation,
        limit: int,
    ) -> tuple[Turn, ...]:
        return tuple(self.turns[conversation.id][-limit:])

    async def load_turns_since_summary(
        self,
        conversation: Conversation,
    ) -> tuple[Turn, ...]:
        real_turns = [
            turn
            for turn in self.turns[conversation.id]
            if not (turn.parts and turn.parts[0].kind == 'summary')
        ]
        boundary_id = conversation.summarized_through_turn_id
        if boundary_id is not None:
            real_turns = [turn for turn in real_turns if turn.id > boundary_id]

        return tuple(sorted(real_turns, key=lambda turn: turn.id))

    async def load_current_summary(
        self,
        conversation: Conversation,
    ) -> Mapping[str, JsonValue] | None:
        summary_turns = [
            turn
            for turn in self.turns[conversation.id]
            if turn.parts and turn.parts[0].kind == 'summary'
        ]
        if not summary_turns:
            return None

        latest = max(summary_turns, key=lambda turn: turn.id)
        return latest.parts[0].payload

    async def apply_compaction(
        self,
        conversation: Conversation,
        summary_payload: Mapping[str, JsonValue],
        turns_folded: tuple[Turn, ...],
        expected_version: int,
    ) -> Conversation | None:
        if not turns_folded:
            raise NoTurnsToCompactError(
                'apply_compaction requires at least one folded turn',
            )

        if self.force_cas_conflict or conversation.compact_version != expected_version:
            return None

        boundary = turns_folded[-1]
        summary_turn = Turn(
            id=self._next_id(),
            parts=(
                Part(
                    role='assistant',
                    kind='summary',
                    payload=cast(Mapping[str, JsonValue], dict(summary_payload)),
                ),
            ),
        )
        self.turns[conversation.id].append(summary_turn)

        updated = replace(
            conversation,
            compact_version=conversation.compact_version + 1,
            summarized_through_turn_id=boundary.id,
        )
        key = (conversation.channel, conversation.chat_id, conversation.agent)
        self.conversations[key] = updated

        return updated


async def test_get_or_create_is_idempotent_per_key() -> None:
    repository = FakeConversationRepository()

    first = await repository.get_or_create('telegram', '42', 'scout')
    again = await repository.get_or_create('telegram', '42', 'scout')

    assert first.id == again.id


async def test_different_agent_starts_a_separate_conversation() -> None:
    repository = FakeConversationRepository()

    scout = await repository.get_or_create('telegram', '42', 'scout')
    sage = await repository.get_or_create('telegram', '42', 'sage')

    assert scout.id != sage.id


async def test_a_turn_holds_ordered_segments_with_their_kinds() -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '42', 'scout')

    turn = await repository.append_turn(
        conversation,
        [
            Part(role='assistant', kind='text', payload={'text': 'checking'}),
            Part(role='assistant', kind='tool_call', payload={'name': 'search'}),
        ],
    )

    assert [(part.role, part.kind) for part in turn.parts] == [
        ('assistant', 'text'),
        ('assistant', 'tool_call'),
    ]


async def test_append_then_load_recent_turns_honours_order_and_limit() -> None:
    repository = FakeConversationRepository()
    conversation = await repository.get_or_create('telegram', '42', 'scout')

    for index in range(3):
        await repository.append_turn(
            conversation,
            [Part(role='user', kind='text', payload={'text': str(index)})],
        )

    turns = await repository.load_recent_turns(conversation, limit=2)

    payloads = [turn.parts[0].payload for turn in turns]
    assert payloads == [{'text': '1'}, {'text': '2'}]
