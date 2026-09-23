from typing import Annotated, override

from pydantic import Field, JsonValue

from lib.decision.exceptions import DecisionResponseError
from lib.decision.types import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResult,
    DecisionUsage,
)

from .base import HTTPDecisionAdapter, WirePayload

# NOTE: Bifrost can answer /v1/decisions by emulating it over chat tool calls on a
# provider that has no native decisions API; that answer has the right shape but is
# not a Jev judgment, so only this upstream is trusted
_JUDGE = 'typesafe'


class BifrostAnswer(WirePayload):
    kind: Annotated[str, Field(pattern='^choice$')]
    value: Annotated[str, Field()]
    confidence: Annotated[float | None, Field()] = None
    probabilities: Annotated[dict[str, float], Field()] = Field(default_factory=dict)


class BifrostUsage(WirePayload):
    prompt_tokens: Annotated[int, Field()] = 0
    completion_tokens: Annotated[int, Field()] = 0


class BifrostExtras(WirePayload):
    provider: Annotated[str | None, Field()] = None


class BifrostResponse(WirePayload):
    model: Annotated[str, Field()]
    answers: Annotated[dict[str, BifrostAnswer], Field()]
    usage: Annotated[BifrostUsage | None, Field()] = None
    extra_fields: Annotated[BifrostExtras | None, Field()] = None


class BifrostAdapter(HTTPDecisionAdapter):
    @override
    def encode(self, request: DecisionRequest) -> dict[str, JsonValue]:
        return {
            'model': request.model,
            'state': request.state,
            'questions': {
                name: {
                    'kind': 'choice',
                    'instructions': question.instructions,
                    'criteria': dict(question.criteria),
                }
                for name, question in request.questions.items()
            },
        }

    @override
    def decode(self, payload: JsonValue) -> DecisionResult:
        response = BifrostResponse.model_validate(payload)
        upstream = (
            None if response.extra_fields is None else response.extra_fields.provider
        )
        if upstream is not None and upstream != _JUDGE:
            raise DecisionResponseError(
                f'{self.provider.name} answered through {upstream!r}, not {_JUDGE!r}',
            )

        return DecisionResult(
            model=response.model,
            answers={
                name: ChoiceAnswer(
                    choice=answer.value,
                    confidence=answer.confidence,
                    probabilities=answer.probabilities,
                )
                for name, answer in response.answers.items()
            },
            usage=(
                None
                if response.usage is None
                else DecisionUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )
            ),
            upstream=upstream,
        )
