from collections.abc import Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue

# NOTE: the TypeSafe limit, enforced upstream with a 400; checking it here turns a
# wasted round trip into a construction error
MAX_CHOICES = 255

type State = str | dict[str, JsonValue] | list[JsonValue]
type Probability = Annotated[float, Field(ge=0, le=1)]


class DecisionValue(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)


class DecisionRef(DecisionValue):
    model: Annotated[str, Field()]
    provider: Annotated[str | None, Field()] = None


class ChoiceQuestion(DecisionValue):
    instructions: Annotated[str, Field(min_length=1)]
    # NOTE: option -> what it means; the option itself is what comes back
    criteria: Annotated[Mapping[str, str], Field(min_length=1, max_length=MAX_CHOICES)]


class DecisionRequest(DecisionValue):
    model: Annotated[str, Field()]
    state: Annotated[State, Field()]
    questions: Annotated[Mapping[str, ChoiceQuestion], Field(min_length=1)]


class ChoiceAnswer(DecisionValue):
    choice: Annotated[str, Field()]
    confidence: Annotated[Probability | None, Field()] = None
    probabilities: Annotated[Mapping[str, Probability], Field()] = Field(
        default_factory=dict,
    )

    @property
    def margin(self) -> float | None:
        ranked = sorted(self.probabilities.values(), reverse=True)
        if not ranked:
            return None

        return ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)


class DecisionUsage(DecisionValue):
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0


class DecisionResult(DecisionValue):
    model: Annotated[str, Field()]
    answers: Annotated[Mapping[str, ChoiceAnswer], Field()]
    usage: Annotated[DecisionUsage | None, Field()] = None
    # NOTE: the upstream a gateway reports having asked, when it says
    upstream: Annotated[str | None, Field()] = None
