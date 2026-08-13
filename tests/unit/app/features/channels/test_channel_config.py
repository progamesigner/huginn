import pytest
from pydantic import SecretStr, ValidationError

from app.core.config.registry import HuginnConfig
from app.features.channels.config import (
    TRANSCRIPTION,
    ChannelConfig,
    ChannelConfigList,
    ChannelRouterConfig,
    TranscriptionConfig,
    _describe_channels,
)
from app.features.channels.exceptions import (
    IncompleteTranscriptionError,
    UnknownChannelKindError,
)
from app.features.channels.registry import ChannelRegistry
from app.features.channels.telegram.config import TelegramChannelConfig


def test_channels_support_multiple_named_telegram_instances() -> None:
    config = ChannelConfigList(
        {
            'support_bot': TelegramChannelConfig(
                kind='telegram',
                token=SecretStr('support-token'),
                secret_token=SecretStr('support-secret'),
                agent='support',
            ),
            'research_bot': TelegramChannelConfig(
                kind='telegram',
                token=SecretStr('research-token'),
                secret_token=SecretStr('research-secret'),
                agent='research',
            ),
        },
    )

    assert config.root['support_bot'].agent == 'support'
    assert config.root['research_bot'].agent == 'research'
    assert 'support-token' not in config.model_dump_json()


def test_channel_schema_rejects_removed_bindings() -> None:
    with pytest.raises(ValidationError, match='bindings'):
        TelegramChannelConfig.model_validate(
            {
                'kind': 'telegram',
                'token': 'token',
                'bindings': {'42': 'scout'},
            },
        )


def test_telegram_channel_requires_token() -> None:
    with pytest.raises(ValidationError):
        TelegramChannelConfig.model_validate({'kind': 'telegram'})


def test_channel_registry_rejects_unknown_kind() -> None:
    registry = ChannelRegistry()
    registry.register('telegram', TelegramChannelConfig)

    with pytest.raises(UnknownChannelKindError, match='discord'):
        registry.get('discord')


def _describe(config: HuginnConfig) -> list[str]:
    assert TRANSCRIPTION.describe is not None

    return TRANSCRIPTION.describe(config)


def _validate(config: HuginnConfig) -> None:
    assert TRANSCRIPTION.validate is not None

    TRANSCRIPTION.validate(config)


def test_an_absent_transcription_section_is_simply_unconfigured() -> None:
    # NOTE: the loader validates a missing section from {}, so voice being unsupported
    # has to be a legal state of the config rather than a load failure
    config = TranscriptionConfig.model_validate({})

    assert config.kind is None
    assert config.model is None
    assert config.api_key is None


def test_a_half_filled_transcription_section_is_refused() -> None:
    config = HuginnConfig(
        sections={'transcription': TranscriptionConfig(kind='openai')},
    )

    with pytest.raises(IncompleteTranscriptionError, match='api_key, model'):
        _validate(config)


def test_a_complete_transcription_section_passes_validation() -> None:
    config = HuginnConfig(
        sections={
            'transcription': TranscriptionConfig(
                kind='openai',
                model='gpt-4o-mini-transcribe',
                api_key=SecretStr('openai-key'),
            ),
        },
    )

    _validate(config)

    assert _describe(config) == ['transcription: openai model=gpt-4o-mini-transcribe']


def test_base_channel_config_is_not_telegram_specific() -> None:
    channel = ChannelConfig(kind='custom', agent='scout')

    assert channel.kind == 'custom'


def test_channel_summary_describes_pinned_and_router_channels() -> None:
    config = HuginnConfig(
        sections={
            'channels': ChannelConfigList(
                {
                    'support_bot': TelegramChannelConfig(
                        kind='telegram',
                        token=SecretStr('support-token'),
                        secret_token=SecretStr('support-secret'),
                        agent='scout',
                    ),
                    'research_bot': TelegramChannelConfig(
                        kind='telegram',
                        token=SecretStr('research-token'),
                        secret_token=SecretStr('research-secret'),
                        agents=('scout', 'concierge'),
                        router=ChannelRouterConfig(
                            fallback='concierge',
                            allowed=('scout', 'concierge'),
                        ),
                    ),
                },
            ),
        },
    )

    assert _describe_channels(config) == [
        'channels: research_bot (telegram, router, 2 candidates), '
        'support_bot (telegram, agent=scout)',
    ]
