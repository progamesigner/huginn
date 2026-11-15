import json
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


class GeminiUsage(WirePayload):
    total_input_tokens: Annotated[int, Field(ge=0)] = 0
    total_output_tokens: Annotated[int, Field(ge=0)] = 0


class GeminiResponse(WirePayload):
    answers: Annotated[dict[str, ChoiceAnswer], Field()]
    usage: Annotated[GeminiUsage | None, Field()] = None


class GeminiAdapter(HTTPDecisionAdapter):
    @override
    def headers(self) -> dict[str, str]:
        return {'x-goog-api-key': self.provider.api_key.get_secret_value()}

    @override
    def encode(self, request: DecisionRequest) -> dict[str, JsonValue]:
        if len(request.questions) > 1000:
            raise DecisionResponseError('Gemini accepts at most 1000 questions')
        if any(len(question.criteria) < 2 for question in request.questions.values()):
            raise DecisionResponseError(
                'Gemini choice questions require at least 2 options',
            )
        return {
            'model': request.model,
            'input': (
                request.state
                if isinstance(request.state, str)
                else json.dumps(request.state, ensure_ascii=False)
            ),
            'questions': {
                name: {
                    'type': 'choice',
                    'instructions': question.instructions,
                    'criteria': dict(question.criteria),
                }
                for name, question in request.questions.items()
            },
        }

    @override
    def decode(self, request: DecisionRequest, payload: JsonValue) -> DecisionResult:
        response = GeminiResponse.model_validate(payload)
        for name, answer in response.answers.items():
            question = request.questions.get(name)
            if question is not None and set(answer.probabilities) != set(
                question.criteria,
            ):
                raise DecisionResponseError('Gemini did not weigh every offered option')
            if abs(sum(answer.probabilities.values()) - 1.0) > 0.01:
                raise DecisionResponseError('Gemini probabilities are not normalized')
        return DecisionResult(
            model=request.model,
            answers=response.answers,
            usage=(
                None
                if response.usage is None
                else DecisionUsage(
                    input_tokens=response.usage.total_input_tokens,
                    output_tokens=response.usage.total_output_tokens,
                )
            ),
        )
