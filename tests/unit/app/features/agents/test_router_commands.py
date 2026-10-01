import json
from pathlib import Path
from typing import override
from unittest.mock import AsyncMock

from typer.testing import CliRunner

from app.core.application import App
from app.core.config.providers import ProviderConfigList
from app.core.config.registry import HuginnConfig
from app.core.providers.command import CommandProvider
from app.features.agents.config import AgentConfigList, DefaultsConfig
from app.features.agents.contracts import TranscriptServiceContract
from app.features.agents.entities import (
    RoutingHistory,
    RoutingPolicy,
    RoutingTarget,
    RoutingTargets,
    SelectionPolicy,
)
from app.features.agents.router import Router, RoutingDecision
from app.features.agents.router_commands import cli
from lib.command import CommandContract
from tests.helpers.logging import make_logger

from .test_agent_router import FakeLLM
from .test_router_evaluation import _sample
from .test_turn_handler import FakeStore


class DatasetStore(FakeStore):
    @override
    async def routing_samples(self, channel, since, history_turns):
        sample = _sample()
        yield RoutingHistory(
            id=sample.id,
            historical_agent=sample.historical_agent,
            context=sample.state.context(),
        )


def test_export_evaluate_review_and_report(tmp_path: Path) -> None:
    container = App().build(CommandProvider())
    config = HuginnConfig(
        {
            'agents': AgentConfigList.model_validate(
                {'a': {'description': 'A'}, 'b': {'description': 'B'}},
            ),
            'defaults': DefaultsConfig(model='chat'),
            'providers': ProviderConfigList.model_validate(
                {
                    'llm': {
                        'adapter': 'openai',
                        'api_key': 'test',
                        'models': ('chat',),
                    },
                    'jev': {
                        'type': 'decision',
                        'adapter': 'typesafe',
                        'api_key': 'test',
                        'models': ('jev',),
                    },
                },
            ),
        },
    )
    container.bind(HuginnConfig, instance=config)
    container.bind(
        RoutingTargets,
        instance=RoutingTargets(
            entries={
                'channels.test': RoutingTarget(
                    channel='test',
                    policy=RoutingPolicy(
                        selection=SelectionPolicy(
                            fallback='a',
                            allowed=frozenset(('a', 'b')),
                        ),
                    ),
                ),
            },
        ),
    )
    container.bind(TranscriptServiceContract, instance=DatasetStore())
    router = Router({}, FakeLLM(), DefaultsConfig(model='chat'), logger=make_logger())
    router.decide = AsyncMock(return_value=RoutingDecision('a'))
    container.bind(Router, instance=router)
    commands = container.resolve(CommandContract)
    commands.mount('router', cli)
    runner = CliRunner()
    dataset = tmp_path / 'dataset.jsonl'
    results = tmp_path / 'results.json'
    labels = tmp_path / 'labels.json'
    output = tmp_path / 'report.json'
    result = runner.invoke(
        commands.app,
        [
            'router',
            'dataset',
            'export',
            '--channel',
            'test',
            '--since',
            '2026-01-01',
            '--out',
            str(dataset),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    result = runner.invoke(
        commands.app,
        [
            'router',
            'eval',
            '--dataset',
            str(dataset),
            '--router',
            'channels.test',
            '--decision-model',
            'jev',
            '--out',
            str(results),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert router.decide.await_count == 3
    result = runner.invoke(
        commands.app,
        [
            'router',
            'dataset',
            'review',
            '--review',
            str(results.with_suffix('.review.jsonl')),
            '--out',
            str(labels),
        ],
        input='a\n',
    )
    assert result.exit_code == 0, (result.output, result.exception)
    result = runner.invoke(
        commands.app,
        [
            'router',
            'eval',
            'report',
            '--dataset',
            str(dataset),
            '--results',
            str(results),
            '--labels',
            str(labels),
            '--out',
            str(output),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(output.read_text())['human_label_count'] == 1
    assert 'continue' not in results.read_text()
    assert 'continue' not in output.read_text()
