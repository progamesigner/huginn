import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.features.agents.entities import (
    RoutingContext,
    RoutingHistory,
    SelectionPolicy,
    TurnPrompt,
)
from app.features.agents.eval_dataset import make_sample, read_samples, write_private
from app.features.agents.eval_report import (
    label_item,
    labels_from_review,
    report,
    review_items,
)
from app.features.agents.evaluation import TokenPrices, Variant, evaluate
from app.features.agents.router import Router, RoutingDecision

from .test_agent_router import FakeLLM


def _sample(identifier: str = '1'):
    now = datetime(2026, 9, 25, tzinfo=UTC)
    return make_sample(
        RoutingHistory(
            identifier,
            'a',
            RoutingContext((TurnPrompt('continue', ''),)),
            now,
            now - timedelta(minutes=1),
        ),
        'channels.test',
        {'a': 'A', 'b': 'B'},
        'a',
    )


def test_export_tags_and_private_file_round_trip(tmp_path: Path) -> None:
    sample = _sample()
    assert set(sample.tags) == {'en', 'short', 'follow-up'}
    path = tmp_path / 'dataset.jsonl'
    write_private(path, sample.model_dump_json() + '\n')
    assert path.stat().st_mode & 0o777 == 0o600
    assert read_samples(path) == (sample,)
    with pytest.raises(FileExistsError):
        write_private(path, 'overwrite')


async def test_three_variants_and_report_do_not_include_state() -> None:
    router = Router({}, FakeLLM())
    router.decide = AsyncMock(
        side_effect=[
            RoutingDecision('a'),
            RoutingDecision('b'),
            RoutingDecision(
                'b',
                confidence=0.9,
                margin=0.8,
                input_tokens=100,
                output_tokens=0,
            ),
        ],
    )
    samples = (_sample(),)
    results = await evaluate(
        router,
        samples,
        {
            'llm': SelectionPolicy('a'),
            'llm+history': SelectionPolicy('a'),
            'decision': SelectionPolicy('a', engine='decision'),
        },
        {'decision': TokenPrices(input_per_million=0.05)},
    )
    assert [result.choice for result in results] == ['a', 'b', 'b']
    assert results[-1].estimated_cost_usd == pytest.approx(0.000005)
    assert 'continue' not in results[-1].model_dump_json()
    review = review_items(samples, results)
    assert review[0].reason == 'disagreement'
    labeled = label_item(review[0], 'b')
    summary = report(samples, results, labels_from_review((labeled,)))
    assert 'continue' not in str(summary)
    assert summary['human_label_count'] == 1
    assert (
        json.loads(json.dumps(summary))['engines']['llm']['human_labels']['accuracy']
        == 0
    )
    assert (
        json.loads(json.dumps(summary))['engines']['decision']['human_labels'][
            'accuracy'
        ]
        == 1
    )
    with pytest.raises(ValueError, match='candidate'):
        label_item(review[0], 'unknown')


async def test_agreement_audit_and_inferred_labels_are_separate() -> None:
    samples = tuple(_sample(str(index)) for index in range(20))
    router = Router({}, FakeLLM())
    router.decide = AsyncMock(return_value=RoutingDecision('a'))
    selections = dict[Variant, SelectionPolicy](
        {
            'llm': SelectionPolicy('a'),
            'llm+history': SelectionPolicy('a'),
            'decision': SelectionPolicy('a', engine='decision'),
        },
    )
    results = await evaluate(router, samples, selections)
    items = review_items(samples, results)
    assert sum(item.reason == 'agreement_audit' for item in items) == 2
    assert sum(item.label_source == 'inferred' for item in items) == 18
    summary = report(samples, results, labels_from_review(items))
    assert summary['human_label_count'] == 0
    assert summary['inferred_label_count'] == 18
    assert (
        json.loads(json.dumps(summary))['engines']['decision']['human_labels'][
            'accuracy'
        ]
        is None
    )
    assert (
        json.loads(json.dumps(summary))['engines']['decision']['estimated_cost_usd']
        is None
    )


async def test_incomplete_comparison_never_infers_labels() -> None:
    router = Router({}, FakeLLM())
    router.decide = AsyncMock(return_value=RoutingDecision('a'))
    samples = (_sample(),)
    results = await evaluate(
        router,
        samples,
        {'decision': SelectionPolicy('a', engine='decision')},
    )
    assert review_items(samples, results)[0].label_source == 'unlabeled'


def test_review_checkpoint_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / 'labels.json'
    write_private(target, 'original')

    def interrupted(*args) -> None:
        raise OSError('interrupted replacement')

    monkeypatch.setattr(os, 'replace', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        write_private(target, 'replacement', overwrite=True)
    assert target.read_text() == 'original'
    assert list(tmp_path.iterdir()) == [target]
