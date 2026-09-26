import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from typer import Context, prompt

from app.core.config.registry import HuginnConfig
from lib.command import Exit, Option, Typer, echo, resolve, scoped

from .config import AGENTS, NO_AGENT, validate_router_selection
from .contracts import TranscriptStoreContract
from .eval_dataset import json_text, make_sample, read_samples, write_private
from .eval_report import (
    Label,
    ReviewItem,
    label_item,
    labels_from_review,
    report,
    review_items,
)
from .evaluation import EvaluationResult, TokenPrices, Variant, evaluate
from .router import Router
from .types import RoutingTarget, RoutingTargets, SelectionPolicy

cli = Typer(no_args_is_help=True, help='Export and evaluate agent routing decisions.')
dataset_cli = Typer(no_args_is_help=True)
eval_cli = Typer(invoke_without_command=True)
cli.add_typer(dataset_cli, name='dataset')
cli.add_typer(eval_cli, name='eval')


def _target(name: str) -> RoutingTarget:
    target = resolve(RoutingTargets).entries.get(name)
    if (
        target is None
        or target.policy.selection is None
        or target.policy.agent is not None
    ):
        raise ValueError('select a router target: channels.<name> or api_keys.<name>')
    return target


@dataset_cli.command('export')
def export(
    since: Annotated[
        datetime,
        Option('--since', formats=['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S%z']),
    ],
    out: Annotated[Path, Option('--out')],
    channel: Annotated[str | None, Option('--channel')] = None,
    api_key: Annotated[str | None, Option('--api-key')] = None,
    history_turns: Annotated[int, Option('--history-turns', min=0)] = 6,
) -> None:
    if (channel is None) == (api_key is None):
        raise ValueError('choose exactly one of --channel and --api-key')
    name = f'channels.{channel}' if channel is not None else f'api_keys.{api_key}'
    target = _target(name)
    if (
        sum(
            other.channel == target.channel
            for other in resolve(RoutingTargets).entries.values()
        )
        > 1
    ):
        raise ValueError(
            'channel and API key share a storage name; historical provenance is ambiguous',
        )
    selection = target.policy.selection
    if selection is None:
        raise ValueError('target has no router')
    agents = resolve(HuginnConfig).section(AGENTS).root
    candidates = {
        key: value.description
        for key, value in agents.items()
        if (not selection.allowed or key in selection.allowed)
        and (not target.policy.allowed or key in target.policy.allowed)
    }
    if selection.fallback not in candidates:
        raise ValueError('fallback must be in the authorized candidate set')

    async def run() -> tuple[int, int]:
        count = followups = 0
        # NOTE: exclusive creation, so a slip of the path cannot overwrite private data
        write_private(out, '')
        try:
            async with scoped() as container:
                store = container.resolve(TranscriptStoreContract)
                with out.open('a') as handle:
                    async for history in store.routing_samples(
                        target.channel,
                        since.replace(tzinfo=UTC) if since.tzinfo is None else since,
                        history_turns,
                    ):
                        sample = make_sample(
                            history,
                            name,
                            candidates,
                            selection.fallback,
                        )
                        handle.write(sample.model_dump_json() + '\n')
                        count += 1
                        followups += int('follow-up' in sample.tags)
        except BaseException:
            out.unlink(missing_ok=True)
            raise
        return count, followups

    count, followups = asyncio.run(run())
    echo(
        f'Exported {count} samples ({followups} follow-ups). Target: 150 / 20. Current-config replay; summaries omitted.',
    )


@eval_cli.callback()
def run_eval(
    context: Context,
    dataset: Annotated[Path | None, Option('--dataset')] = None,
    router: Annotated[str | None, Option('--router')] = None,
    out: Annotated[Path | None, Option('--out')] = None,
    engine: Annotated[
        Literal['llm', 'llm+history', 'decision', 'both'],
        Option('--engine'),
    ] = 'both',
    decision_model: Annotated[str | None, Option('--decision-model')] = None,
    decision_provider: Annotated[str | None, Option('--decision-provider')] = None,
    llm_model: Annotated[str | None, Option('--llm-model')] = None,
    llm_provider: Annotated[str | None, Option('--llm-provider')] = None,
    prices: Annotated[
        Path | None,
        Option(
            '--prices',
            help='JSON keyed by variant with input_per_million and output_per_million USD.',
        ),
    ] = None,
) -> None:
    if context.invoked_subcommand is not None:
        return
    if dataset is None or router is None or out is None:
        echo('--dataset, --router and --out are required', err=True)
        raise Exit(2)
    samples = read_samples(dataset)
    target = _target(router)
    selection = target.policy.selection
    if selection is None:
        raise ValueError('target has no router')
    config = resolve(HuginnConfig)
    if any(
        sample.router != router
        or sample.fallback != selection.fallback
        or not set(sample.candidates).issubset(selection.allowed)
        or (
            target.policy.allowed
            and not set(sample.candidates).issubset(target.policy.allowed)
        )
        for sample in samples
    ):
        raise ValueError(
            'dataset router, fallback or candidates differ from current authorized configuration; export again',
        )
    llm = SelectionPolicy(
        selection.fallback,
        model=llm_model or (selection.model if selection.engine == 'llm' else None),
        provider=llm_provider
        or (selection.provider if selection.engine == 'llm' else None),
    )
    decision = SelectionPolicy(
        selection.fallback,
        engine='decision',
        model=decision_model
        or (selection.model if selection.engine == 'decision' else None),
        provider=decision_provider
        or (selection.provider if selection.engine == 'decision' else None),
    )
    choices = dict[Variant, SelectionPolicy](
        {
            'llm': llm,
            'llm+history': llm,
            'decision': decision,
        },
    )
    selections = choices if engine == 'both' else {engine: choices[engine]}
    for choice in selections.values():
        validate_router_selection(
            config,
            router,
            engine=choice.engine,
            provider=choice.provider,
            model=choice.model,
            allowed=tuple(selection.allowed),
            thresholds=False,
        )
    price_map = (
        {
            key: TokenPrices.model_validate(value)
            for key, value in json.loads(prices.read_text()).items()
        }
        if prices
        else {}
    )
    review_path = out.with_suffix('.review.jsonl')
    if out.exists() or review_path.exists():
        raise FileExistsError('result or review output already exists')

    async def run() -> tuple[EvaluationResult, ...]:
        async with scoped() as container:
            return await evaluate(
                container.resolve(Router),
                samples,
                selections,
                price_map,
            )

    results = asyncio.run(run())
    write_private(out, json_text([result.model_dump() for result in results]))
    items = review_items(samples, results)
    write_private(review_path, ''.join(item.model_dump_json() + '\n' for item in items))
    echo(
        f'Evaluated {len(samples)} samples, {len(results)} decisions. Review: {review_path}',
    )


@dataset_cli.command('review')
def review(
    source: Annotated[Path, Option('--review')],
    out: Annotated[
        Path,
        Option('--out', help='Labels JSON, without conversation text.'),
    ],
) -> None:
    if source.resolve() == out.resolve():
        raise ValueError('review and labels must use different files')
    items = [
        ReviewItem.model_validate_json(line)
        for line in source.read_text().splitlines()
        if line.strip()
    ]
    for index, item in enumerate(items):
        if item.label_source != 'unlabeled':
            continue
        echo(
            json_text(
                {
                    'id': item.id,
                    'state': item.state.model_dump(),
                    'choices': item.choices,
                    'historical_agent': item.historical_agent,
                },
            ),
        )
        while True:
            answer = prompt(
                f'Expected ({", ".join(item.candidates)}; skip / quit)',
                default='skip',
            )
            if answer in ('skip', 'quit'):
                break
            try:
                items[index] = label_item(item, answer)
                break
            except ValueError:
                echo(f'Choose a listed agent or {NO_AGENT}.')
        write_private(
            source,
            ''.join(entry.model_dump_json() + '\n' for entry in items),
            overwrite=True,
        )
        write_private(
            out,
            json_text(
                [label.model_dump() for label in labels_from_review(tuple(items))],
            ),
            overwrite=True,
        )
        if answer == 'quit':
            return
    write_private(
        out,
        json_text([label.model_dump() for label in labels_from_review(tuple(items))]),
        overwrite=True,
    )


@eval_cli.command('report')
def evaluation_report(
    dataset: Annotated[Path, Option('--dataset')],
    results: Annotated[Path, Option('--results')],
    labels: Annotated[Path, Option('--labels')],
    out: Annotated[Path, Option('--out')],
) -> None:
    summary = report(
        read_samples(dataset),
        tuple(
            EvaluationResult.model_validate(item)
            for item in json.loads(results.read_text())
        ),
        tuple(Label.model_validate(item) for item in json.loads(labels.read_text())),
    )
    write_private(out, json_text(summary))
    echo(f'Report written to {out}')
