import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lib.llm import (
    CompletionRequest,
    LLMClient,
    LLMContract,
    Message,
    OutputSchema,
    TextChunk,
    TextSegment,
)

from .exceptions import NoTurnsToCompactError
from .repositories import ConversationRepository
from .types import Conversation, Part, Turn

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_DEFAULT_KEEP_TURNS = 8
_DEFAULT_TOKEN_THRESHOLD = 50_000
_DEFAULT_MAX_RETRIES = 1
_TEXT = 'text'
_REPLAYED = ('audio', 'image', _TEXT)


class SummaryOutput(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra='forbid')

    summary: Annotated[str, Field(min_length=1)]
    key_facts: Annotated[tuple[str, ...], Field()] = ()
    open_items: Annotated[tuple[str, ...], Field()] = ()


@dataclass(frozen=True)
class CompactionResult:
    compacted: bool
    conversation: Conversation


def estimate_tokens(turns: Sequence[Turn]) -> int:
    chars = sum(
        len(json.dumps(dict(part.payload), default=str))
        for turn in turns
        for part in turn.parts
    )
    return max(1, chars // _CHARS_PER_TOKEN)


_INSTRUCTION = (
    'Summarize the conversation above for future context. Respond with only a JSON '
    'object matching this shape: {"summary": "<narrative recap>", "key_facts": '
    '["<durable fact or decision>", ...], "open_items": ["<unresolved question or '
    'next step>", ...]}. key_facts and open_items may be empty arrays if nothing '
    'durable applies.'
)


def _text_of(part: Part) -> str | None:
    if part.kind not in _REPLAYED:
        return None

    text = part.payload.get('text')
    return text if isinstance(text, str) and text else None


def replay_messages(turns: Sequence[Turn]) -> Iterator[Message]:
    for turn in turns:
        for part in turn.parts:
            # NOTE: role is a constrained column, not a literal type, so replaying it
            # into a model turn has to narrow it back to the two roles that can speak
            if (text := _text_of(part)) is None:
                continue

            if part.role == 'assistant':
                yield Message(role='assistant', segments=(TextSegment(text=text),))
            elif part.role == 'user':
                yield Message(role='user', segments=(TextSegment(text=text),))


def summary_message(summary: SummaryOutput) -> Message:
    return Message(
        role='user',
        segments=(
            TextSegment(text=f'Summary of the conversation so far: {summary.summary}'),
        ),
        cacheable=True,
    )


def _history_messages(
    prior_summary: SummaryOutput | None,
    turns: Sequence[Turn],
) -> Iterator[Message]:
    if prior_summary is not None:
        yield summary_message(prior_summary)
    yield from replay_messages(turns)


def _fold_instruction(prior_summary: SummaryOutput | None, instruction: str) -> str:
    if prior_summary is None:
        return instruction

    return (
        'An earlier summary of even older turns already exists — fold it in rather '
        'than discarding it:\n'
        f'Known facts: {"; ".join(prior_summary.key_facts) or "none"}\n'
        f'Open items: {"; ".join(prior_summary.open_items) or "none"}\n\n'
        f'{instruction}'
    )


def _summary_messages(
    instructions: str,
    prior_summary: SummaryOutput | None,
    turns_to_fold: tuple[Turn, ...],
    instruction: str,
) -> Iterator[Message]:
    system = Message(role='system', segments=(TextSegment(text=instructions),))
    history = list(_history_messages(prior_summary, turns_to_fold))

    if history:
        history[-1] = history[-1].model_copy(update={'cacheable': True})
    else:
        system = system.model_copy(update={'cacheable': True})

    yield system
    yield from history
    yield Message(
        role='user',
        segments=(TextSegment(text=_fold_instruction(prior_summary, instruction)),),
    )


class CompactionService:
    llm: LLMContract
    repository: ConversationRepository
    keep_turns: int
    token_threshold: int
    max_retries: int
    instructions: str
    instruction: str

    def __init__(
        self,
        llm: LLMContract,
        repository: ConversationRepository,
        keep_turns: int = _DEFAULT_KEEP_TURNS,
        token_threshold: int = _DEFAULT_TOKEN_THRESHOLD,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        instructions: str = '',
        instruction: str | None = None,
    ) -> None:
        self.llm = llm
        self.repository = repository
        self.keep_turns = keep_turns
        self.instructions = instructions
        self.token_threshold = token_threshold
        self.max_retries = max_retries
        self.instruction = instruction or _INSTRUCTION

    async def maybe_compact(
        self,
        conversation: Conversation,
        model: str,
    ) -> CompactionResult:
        real_turns = await self.repository.load_turns_since_summary(conversation)

        if len(real_turns) <= self.keep_turns:
            return CompactionResult(compacted=False, conversation=conversation)

        foldable = real_turns[: -self.keep_turns]
        if not self.should_compact(foldable):
            return CompactionResult(compacted=False, conversation=conversation)

        return await self.compact(
            conversation,
            await self._prior_summary(conversation),
            foldable,
            model,
        )

    async def force(
        self,
        conversation: Conversation,
        model: str,
    ) -> CompactionResult:
        foldable = await self.repository.load_turns_since_summary(conversation)
        if not foldable:
            return CompactionResult(compacted=False, conversation=conversation)

        return await self.compact(
            conversation,
            await self._prior_summary(conversation),
            foldable,
            model,
        )

    async def _prior_summary(
        self,
        conversation: Conversation,
    ) -> SummaryOutput | None:
        payload = await self.repository.load_current_summary(conversation)
        if payload is None:
            return None

        return SummaryOutput.model_validate(payload, strict=False)

    def should_compact(self, foldable: tuple[Turn, ...]) -> bool:
        if not foldable:
            return False

        return (
            len(foldable) >= self.keep_turns
            or estimate_tokens(foldable) >= self.token_threshold
        )

    async def compact(
        self,
        conversation: Conversation,
        prior_summary: SummaryOutput | None,
        turns_to_fold: tuple[Turn, ...],
        model: str,
    ) -> CompactionResult:
        if not turns_to_fold:
            raise NoTurnsToCompactError('compact requires at least one turn to fold')

        client = self.llm.client_for(model)
        if client is None:
            return CompactionResult(compacted=False, conversation=conversation)

        output = await self._summarize(prior_summary, turns_to_fold, model, client)
        if output is None:
            return CompactionResult(compacted=False, conversation=conversation)

        updated = await self.repository.apply_compaction(
            conversation,
            summary_payload=output.model_dump(mode='json'),
            turns_folded=turns_to_fold,
            expected_version=conversation.compact_version,
        )
        if updated is None:
            return CompactionResult(compacted=False, conversation=conversation)

        return CompactionResult(compacted=True, conversation=updated)

    async def _summarize(
        self,
        prior_summary: SummaryOutput | None,
        turns_to_fold: tuple[Turn, ...],
        model: str,
        client: LLMClient,
    ) -> SummaryOutput | None:
        messages = tuple(
            _summary_messages(
                self.instructions,
                prior_summary,
                turns_to_fold,
                self.instruction,
            ),
        )

        for attempt in range(self.max_retries + 1):
            request = CompletionRequest(
                model=model,
                messages=messages,
                output_schema=OutputSchema.from_model(SummaryOutput),
            )
            fragments = []
            async for chunk in client.stream(request):
                match chunk:
                    case TextChunk(text=fragment):
                        fragments.append(fragment)
                    case _:
                        pass
            raw = ''.join(fragments)

            try:
                return SummaryOutput.model_validate_json(raw)
            except ValidationError:
                logger.warning(
                    'compact: summary output failed validation on attempt %d/%d',
                    attempt + 1,
                    self.max_retries + 1,
                )
                continue

        return None
