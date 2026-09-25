import json
import os
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .routing_context import routing_state
from .types import RoutingContext, RoutingHistory, RoutingTurn, TurnPrompt


# NOTE: replay data comes from personal conversations; it stays on this machine and
# never reaches a log
class EvalValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid', hide_input_in_errors=True)


class ReplayTurn(EvalValue):
    handled_by: str
    user: tuple[str, ...]
    assistant: str
    attachments: tuple[str, ...] = ()


class ReplayState(EvalValue):
    current: tuple[str, ...]
    previous_turns: tuple[ReplayTurn, ...] = ()
    summary: str | None = None

    def context(self, history: bool = True) -> RoutingContext:
        return RoutingContext(
            current=tuple(TurnPrompt(text=text, actor='') for text in self.current),
            turns=tuple(
                RoutingTurn(
                    turn.handled_by,
                    turn.user,
                    turn.assistant,
                    turn.attachments,
                )
                for turn in self.previous_turns
            )
            if history
            else (),
            summary=self.summary if history else None,
        )


class DatasetSample(EvalValue):
    version: Annotated[int, Field(ge=1, le=1)] = 1
    id: str
    router: str
    state: ReplayState
    legacy_message: str
    candidates: Annotated[dict[str, str], Field(min_length=1, max_length=254)]
    fallback: str
    historical_agent: str
    tags: tuple[str, ...]
    provenance: str = (
        'current_config_replay; historical routing mode unknown; summaries omitted'
    )


def make_sample(
    history: RoutingHistory,
    router: str,
    candidates: dict[str, str],
    fallback: str,
) -> DatasetSample:
    text = '\n'.join(prompt.text for prompt in history.context.current)
    letters = [char for char in text if char.isalpha()]
    chinese = sum('\u4e00' <= char <= '\u9fff' for char in letters)
    tags = ['zh' if chinese > len(letters) / 2 else 'en']
    if len(text) < 15:
        tags.append('short')
    if '[attachments:' in text or any(
        prompt.attachments for prompt in history.context.current
    ):
        tags.append('attachment')
    if history.created_at and history.previous_at:
        if (
            timedelta(0)
            <= history.created_at - history.previous_at
            < timedelta(minutes=30)
        ):
            tags.append('follow-up')
    return DatasetSample(
        id=history.id,
        router=router,
        state=ReplayState.model_validate(routing_state(history.context)),
        legacy_message=history.context.current[-1].text
        if history.context.current
        else '',
        candidates=candidates,
        fallback=fallback,
        historical_agent=history.historical_agent,
        tags=tuple(tags),
    )


def read_samples(path: Path) -> tuple[DatasetSample, ...]:
    samples = tuple(
        DatasetSample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError('duplicate dataset ids')
    return samples


def write_private(path: Path, content: str, *, overwrite: bool = False) -> None:
    # NOTE: review files come from the same conversations, so every file is private
    if not overwrite:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(content)
        return

    with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
