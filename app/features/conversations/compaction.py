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
from .types import Conversation, Turn

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_DEFAULT_KEEP_TURNS = 8
_DEFAULT_TOKEN_THRESHOLD = 50_000
_DEFAULT_MAX_RETRIES = 1


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


def _transcript(
    prior_summary: SummaryOutput | None,
    turns_to_fold: tuple[Turn, ...],
) -> Iterator[str]:
    if prior_summary is not None:
        yield (
            'An earlier summary of even older turns already exists — fold it in rather '
            'than discarding it:\n'
            f'Summary so far: {prior_summary.summary}\n'
            f'Known facts: {"; ".join(prior_summary.key_facts) or "none"}\n'
            f'Open items: {"; ".join(prior_summary.open_items) or "none"}'
        )
    for turn in turns_to_fold:
        for part in turn.parts:
            yield f'[{part.role}/{part.kind}] {json.dumps(dict(part.payload))}'


def _summary_messages(
    prior_summary: SummaryOutput | None,
    turns_to_fold: tuple[Turn, ...],
) -> Iterator[Message]:
    transcript = '\n\n'.join(_transcript(prior_summary, turns_to_fold))
    yield Message(
        role='user',
        segments=(TextSegment(text=transcript),),
        cacheable=True,
    )
    yield Message(role='user', segments=(TextSegment(text=_INSTRUCTION),))


class CompactionService:
    llm: LLMContract
    repository: ConversationRepository
    keep_turns: int
    token_threshold: int
    max_retries: int

    def __init__(
        self,
        llm: LLMContract,
        repository: ConversationRepository,
        keep_turns: int = _DEFAULT_KEEP_TURNS,
        token_threshold: int = _DEFAULT_TOKEN_THRESHOLD,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self.llm = llm
        self.repository = repository
        self.keep_turns = keep_turns
        self.token_threshold = token_threshold
        self.max_retries = max_retries

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

        prior_payload = await self.repository.load_current_summary(conversation)
        prior_summary = (
            SummaryOutput.model_validate(prior_payload, strict=False)
            if prior_payload is not None
            else None
        )
        return await self.compact(conversation, prior_summary, foldable, model)

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
        messages = tuple(_summary_messages(prior_summary, turns_to_fold))

        for attempt in range(self.max_retries + 1):
            request = CompletionRequest(
                model=model,
                messages=messages,
                output_schema=OutputSchema.from_model(SummaryOutput),
            )
            fragments: list[str] = []
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
