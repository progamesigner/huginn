from typing import Annotated, override

from pydantic import Field, JsonValue

from lib.decision.types import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResult,
    DecisionUsage,
)

from .base import HTTPDecisionAdapter, WirePayload


class NativeAnswer(WirePayload):
    type: Annotated[str, Field(pattern='^choice$')]
    choice: Annotated[str, Field()]
    confidence: Annotated[float | None, Field()] = None
    probabilities: Annotated[dict[str, float], Field()] = Field(default_factory=dict)


class NativeUsage(WirePayload):
    input_tokens: Annotated[int, Field()] = 0
    output_tokens: Annotated[int, Field()] = 0


class NativeResponse(WirePayload):
    model: Annotated[str, Field()]
    answers: Annotated[dict[str, NativeAnswer], Field()]
    usage: Annotated[NativeUsage | None, Field()] = None


def encode_native(request: DecisionRequest) -> dict[str, JsonValue]:
    return {
        'model': request.model,
        'state': request.state,
        'questions': {
            name: {
                'type': 'choice',
                'instructions': question.instructions,
                'criteria': dict(question.criteria),
            }
            for name, question in request.questions.items()
        },
    }


def decode_native(payload: JsonValue) -> DecisionResult:
    response = NativeResponse.model_validate(payload)
    return DecisionResult(
        model=response.model,
        answers={
            name: ChoiceAnswer(
                choice=answer.choice,
                confidence=answer.confidence,
                probabilities=answer.probabilities,
            )
            for name, answer in response.answers.items()
        },
        usage=(
            None
            if response.usage is None
            else DecisionUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        ),
    )


class TypeSafeAdapter(HTTPDecisionAdapter):
    @override
    def encode(self, request: DecisionRequest) -> dict[str, JsonValue]:
        return encode_native(request)

    @override
    def decode(self, payload: JsonValue) -> DecisionResult:
        return decode_native(payload)
