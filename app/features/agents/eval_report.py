import hashlib
import math
from collections import Counter
from typing import Literal

from .config import NO_AGENT
from .eval_dataset import DatasetSample, EvalValue, ReplayState
from .evaluation import EvaluationResult


class ReviewItem(EvalValue):
    id: str
    state: ReplayState
    candidates: tuple[str, ...]
    choices: dict[str, str | None]
    historical_agent: str
    reason: Literal['disagreement', 'agreement_audit', 'inferred']
    expected: str | None = None
    label_source: Literal['human', 'inferred', 'unlabeled'] = 'unlabeled'


class Label(EvalValue):
    id: str
    expected: str
    source: Literal['human', 'inferred']
    reason: str


def review_items(
    samples: tuple[DatasetSample, ...],
    results: tuple[EvaluationResult, ...],
) -> tuple[ReviewItem, ...]:
    by_id = dict[str, dict[str, EvaluationResult]]()
    for result in results:
        by_id.setdefault(result.id, {})[result.engine] = result
    agreed = []
    for sample in samples:
        entries = by_id.get(sample.id, {})
        if (
            set(entries) == {'llm', 'llm+history', 'decision'}
            and all(
                result.fallback_reason is None
                and result.choice == sample.historical_agent
                for result in entries.values()
            )
            and sample.historical_agent in sample.candidates
        ):
            agreed.append(sample.id)
    # NOTE: ordering by digest makes the audit sample stable whatever the input order,
    # and it takes exactly ceil(10%) of the agreements
    audit = set(
        sorted(agreed, key=lambda value: hashlib.sha256(value.encode()).digest())[
            : math.ceil(len(agreed) / 10)
        ],
    )
    items = []
    for sample in samples:
        inferred = sample.id in agreed and sample.id not in audit
        items.append(
            ReviewItem(
                id=sample.id,
                state=sample.state,
                candidates=(*sample.candidates, NO_AGENT),
                choices={
                    engine: result.choice
                    for engine, result in by_id.get(sample.id, {}).items()
                },
                historical_agent=sample.historical_agent,
                reason='inferred'
                if inferred
                else ('agreement_audit' if sample.id in audit else 'disagreement'),
                expected=sample.historical_agent if inferred else None,
                label_source='inferred' if inferred else 'unlabeled',
            ),
        )
    return tuple(items)


def label_item(item: ReviewItem, expected: str) -> ReviewItem:
    if expected not in item.candidates:
        raise ValueError(f'expected must be a candidate or {NO_AGENT}')
    return item.model_copy(update={'expected': expected, 'label_source': 'human'})


def labels_from_review(items: tuple[ReviewItem, ...]) -> tuple[Label, ...]:
    return tuple(
        Label(
            id=item.id,
            expected=item.expected,
            source=item.label_source,
            reason=item.reason,
        )
        for item in items
        if item.expected is not None and item.label_source in ('human', 'inferred')
    )


def percentile(values: list[float], fraction: float) -> float | None:
    return (
        sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]
        if values
        else None
    )


def _accuracy(
    results: list[EvaluationResult],
    labels: dict[str, Label],
) -> dict[str, object]:
    labeled = [result for result in results if result.id in labels]
    correct = sum(result.choice == labels[result.id].expected for result in labeled)
    return {
        'count': len(labeled),
        'correct': correct,
        'accuracy': correct / len(labeled) if labeled else None,
    }


def report(
    samples: tuple[DatasetSample, ...],
    results: tuple[EvaluationResult, ...],
    labels: tuple[Label, ...],
) -> dict[str, object]:
    sample_by_id = {sample.id: sample for sample in samples}
    if len({(result.id, result.engine) for result in results}) != len(results):
        raise ValueError('duplicate evaluation results')
    if len({label.id for label in labels}) != len(labels):
        raise ValueError('duplicate labels')
    if any(result.id not in sample_by_id for result in results) or any(
        label.id not in sample_by_id for label in labels
    ):
        raise ValueError('results and labels must reference dataset ids')
    for label in labels:
        if label.expected not in (*sample_by_id[label.id].candidates, NO_AGENT):
            raise ValueError('label is outside the candidate set')
    human = {label.id: label for label in labels if label.source == 'human'}
    inferred = {label.id: label for label in labels if label.source == 'inferred'}
    engines = dict[str, object]()
    for engine in sorted({result.engine for result in results}):
        group = [result for result in results if result.engine == engine]
        confusion = dict[str, dict[str, int]]()
        for result in group:
            if result.id in human:
                row = confusion.setdefault(human[result.id].expected, {})
                predicted = result.choice or '__error__'
                row[predicted] = row.get(predicted, 0) + 1
        bins = dict[str, object]()
        for metric in ('confidence', 'margin'):
            buckets = []
            for low in (0.0, 0.2, 0.4, 0.6, 0.8):
                subset = [
                    result
                    for result in group
                    if (value := getattr(result, metric)) is not None
                    and low <= value
                    and (value < low + 0.2 or low == 0.8)
                ]
                buckets.append(
                    {
                        'low': low,
                        'high': round(low + 0.2, 1),
                        'total': len(subset),
                        **_accuracy(subset, human),
                    },
                )
            bins[metric] = buckets
        costs = [
            result.estimated_cost_usd
            for result in group
            if result.estimated_cost_usd is not None
        ]
        engines[engine] = {
            'count': len(group),
            'human_labels': _accuracy(group, human),
            'inferred_labels': _accuracy(group, inferred),
            'by_tag': {
                tag: _accuracy(
                    [result for result in group if tag in sample_by_id[result.id].tags],
                    human,
                )
                for tag in sorted({tag for sample in samples for tag in sample.tags})
            },
            'confusion_matrix_human': confusion,
            'fallback_rate': sum(result.fallback_reason is not None for result in group)
            / len(group),
            'fallback_reasons': dict(
                Counter(
                    result.fallback_reason for result in group if result.fallback_reason
                ),
            ),
            'latency_ms': {
                'p50': percentile([result.elapsed_ms for result in group], 0.5),
                'p95': percentile([result.elapsed_ms for result in group], 0.95),
            },
            'estimated_cost_usd': sum(costs) if costs else None,
            'cost_coverage': len(costs) / len(group),
            'calibration_human': bins,
        }
    audited = [label for label in human.values() if label.reason == 'agreement_audit']
    return {
        'sample_count': len(samples),
        'follow_up_count': sum('follow-up' in sample.tags for sample in samples),
        'human_label_count': len(human),
        'inferred_label_count': len(inferred),
        'agreement_audit_count': len(audited),
        'agreement_audit_error_rate': sum(
            label.expected != sample_by_id[label.id].historical_agent
            for label in audited
        )
        / len(audited)
        if audited
        else None,
        'limitations': [
            'Current configuration replay; historical candidate and explicit-agent choices were not recorded.',
            'Historical summaries omitted to prevent future leakage.',
            'Human labels oversample disagreements; subset accuracy is not population accuracy.',
            'Inferred labels measure agreement, not independently verified correctness.',
            'LLM token counts use characters / 4; costs require supplied per-model prices.',
        ],
        'recommendation': 'Do not roll out until human review and sample coverage are sufficient.',
        'engines': engines,
    }
