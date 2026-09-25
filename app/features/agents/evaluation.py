from dataclasses import asdict
from typing import Annotated, Literal

from pydantic import Field

from .config import AgentConfig
from .eval_dataset import DatasetSample, EvalValue
from .router import Router, _prompt
from .routing_context import render_routing_state
from .types import RoutingContext, SelectionPolicy, TurnPrompt

type Variant = Literal['llm', 'llm+history', 'decision']
VARIANTS: tuple[Variant, ...] = ('llm', 'llm+history', 'decision')


class TokenPrices(EvalValue):
    input_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    output_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0


class EvaluationResult(EvalValue):
    id: str
    engine: Variant
    choice: str | None
    effective_agent: str
    fallback_reason: str | None
    confidence: float | None
    margin: float | None
    model: str | None
    upstream: str | None
    elapsed_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None = None
    usage_basis: str = 'unavailable'


async def evaluate(
    router: Router,
    samples: tuple[DatasetSample, ...],
    selections: dict[Variant, SelectionPolicy],
    prices: dict[str, TokenPrices] | None = None,
) -> tuple[EvaluationResult, ...]:
    results = []
    for sample in samples:
        candidates = {
            name: AgentConfig(description=description)
            for name, description in sample.candidates.items()
        }
        for variant, selection in selections.items():
            context = (
                RoutingContext((TurnPrompt(sample.legacy_message, ''),))
                if variant == 'llm'
                else sample.state.context()
            )
            decision = await router.decide(selection, context, candidates)
            fields = asdict(decision)
            cost = None
            basis = 'reported' if decision.input_tokens is not None else 'unavailable'
            rate = (prices or {}).get(variant)
            if rate is not None and decision.input_tokens is not None:
                cost = (
                    decision.input_tokens * rate.input_per_million
                    + (decision.output_tokens or 0) * rate.output_per_million
                ) / 1_000_000
            # NOTE: the LLM port streams text without usage, so this is a character
            # estimate, labelled apart from reported tokens and billing
            if variant != 'decision' and decision.choice is not None:
                input_chars = len(_prompt(candidates, router.prompt)) + len(
                    render_routing_state(context),
                )
                fields['input_tokens'] = max(1, (input_chars + 3) // 4)
                fields['output_tokens'] = max(1, (len(decision.choice) + 3) // 4)
                basis = 'estimated_chars_div_4'
                if rate is not None:
                    cost = (
                        fields['input_tokens'] * rate.input_per_million
                        + fields['output_tokens'] * rate.output_per_million
                    ) / 1_000_000
            results.append(
                EvaluationResult(
                    id=sample.id,
                    engine=variant,
                    **fields,
                    effective_agent=decision.choice
                    if decision.fallback_reason is None and decision.choice
                    else sample.fallback,
                    estimated_cost_usd=cost,
                    usage_basis=basis,
                ),
            )
    return tuple(results)
