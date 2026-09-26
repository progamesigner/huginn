import json

from typer.testing import CliRunner

from app.features.agents.router_commands import cli
from app.features.agents.shadow_report import (
    ShadowRecord,
    shadow_records,
    shadow_report,
)


def _line(
    choice: str,
    primary: str,
    confidence: str = '0.8',
    reason: str = 'None',
    elapsed: str = '120.0',
) -> str:
    return (
        f'Router shadow engine=decision choice={choice} primary={primary} '
        f'agree={choice == primary} confidence={confidence} margin=None '
        f'elapsed_ms={elapsed} model=jev upstream=None reason={reason} chat=42'
    )


def test_records_are_read_from_json_and_console_lines_alike() -> None:
    lines = [
        json.dumps({'event': _line('a', 'a'), 'level': 'info'}),
        _line('b', 'a', confidence='0.4'),
        json.dumps({'event': 'Router engine=llm choice=a', 'level': 'info'}),
        'not a log line at all',
    ]

    assert shadow_records(lines) == (
        ShadowRecord('a', 'a', 0.8, 120.0, None),
        ShadowRecord('a', 'b', 0.4, 120.0, None),
    )


def test_the_report_splits_agreement_by_agent_and_confidence() -> None:
    records = shadow_records(
        [
            _line('a', 'a', confidence='0.95'),
            _line('a', 'a', confidence='0.75'),
            _line('b', 'a', confidence='0.4'),
            _line('b', 'b', confidence='1.0'),
            _line('None', 'a', confidence='None', reason='DecisionTimeoutError'),
        ],
    )

    report = shadow_report(records)

    assert report['shadowed'] == 5
    assert report['answered'] == {'count': 4, 'agreed': 3, 'agreement': 0.75}
    assert report['fallbacks'] == {'DecisionTimeoutError': 1}
    assert report['by_primary'] == {
        'a': {'count': 3, 'agreed': 2, 'agreement': 2 / 3},
        'b': {'count': 1, 'agreed': 1, 'agreement': 1.0},
    }
    assert report['by_confidence'] == {
        '0.0-0.5': {'count': 1, 'agreed': 0, 'agreement': 0.0},
        '0.5-0.7': {'count': 0, 'agreed': 0, 'agreement': None},
        '0.7-0.9': {'count': 1, 'agreed': 1, 'agreement': 1.0},
        '0.9-1.0': {'count': 2, 'agreed': 2, 'agreement': 1.0},
    }
    assert report['disagreements'] == {'a -> b': 1}


def test_the_command_reads_a_log_from_standard_input() -> None:
    result = CliRunner().invoke(
        cli,
        ['shadow', 'report', '--log', '-'],
        input=_line('a', 'a') + '\n' + _line('b', 'a') + '\n',
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['answered']['agreement'] == 0.5
