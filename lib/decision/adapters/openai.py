import json
from typing import Annotated, Literal, override

from pydantic import Field, JsonValue

from lib.decision.exceptions import DecisionResponseError
from lib.decision.types import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResult,
    DecisionUsage,
)

from .base import HTTPDecisionAdapter, WirePayload


class OpenAIProbability(WirePayload):
    value: Annotated[str, Field()]
    probability: Annotated[float, Field()]


class OpenAIChoice(WirePayload):
    type: Annotated[Literal['choice'], Field()]
    name: Annotated[str | None, Field()] = None
    choice: Annotated[str, Field()]
    confidence: Annotated[float | None, Field()] = None
    probabilities: Annotated[list[OpenAIProbability], Field()] = Field(
        default_factory=list,
    )


class OpenAIRefusal(WirePayload):
    type: Annotated[Literal['refusal'], Field()]
    name: Annotated[str | None, Field()] = None


type OpenAIAnswer = Annotated[
    OpenAIChoice | OpenAIRefusal,
    Field(discriminator='type'),
]


class OpenAIUsage(WirePayload):
    input_tokens: Annotated[int, Field()] = 0
    output_tokens: Annotated[int, Field()] = 0


class OpenAIResponse(WirePayload):
    model: Annotated[str, Field()]
    answers: Annotated[list[OpenAIAnswer], Field()]
    usage: Annotated[OpenAIUsage | None, Field()] = None


class OpenAIAdapter(HTTPDecisionAdapter):
    @override
    def encode(self, request: DecisionRequest) -> dict[str, JsonValue]:
        return {
            'model': request.model,
            # NOTE: OpenAI takes text or user messages only, so structured state goes
            # over as its JSON text
            'input': (
                request.state
                if isinstance(request.state, str)
                else json.dumps(request.state, ensure_ascii=False)
            ),
            'questions': [
                {
                    'type': 'choice',
                    'name': name,
                    'instructions': question.instructions,
                    'choices': [
                        {'value': value, 'description': description}
                        for value, description in question.criteria.items()
                    ],
                }
                for name, question in request.questions.items()
            ],
        }

    @override
    def decode(self, request: DecisionRequest, payload: JsonValue) -> DecisionResult:
        response = OpenAIResponse.model_validate(payload)

        answers = {}
        asked_names = tuple(request.questions)
        if len(response.answers) != len(asked_names):
            raise DecisionResponseError(
                f'{self.provider.name} returned an unexpected number of answers',
            )

        # NOTE: answers come back in question order; when a name is echoed it must
        # match that position, and unnamed answers use the position as their key
        for asked, answer in zip(asked_names, response.answers, strict=True):
            if answer.name is not None and answer.name != asked:
                raise DecisionResponseError(
                    f'{self.provider.name} answered question {asked!r} '
                    f'as {answer.name!r}',
                )
            name = asked
            if isinstance(answer, OpenAIRefusal):
                raise DecisionResponseError(f'question {name!r} was refused')
            answers[name] = ChoiceAnswer(
                choice=answer.choice,
                confidence=answer.confidence,
                probabilities={
                    weighed.value: weighed.probability
                    for weighed in answer.probabilities
                },
            )

        return DecisionResult(
            model=response.model,
            answers=answers,
            usage=(
                None
                if response.usage is None
                else DecisionUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            ),
        )
