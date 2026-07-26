import pytest
from pydantic import SecretStr, ValidationError

from app.features.channels.config import ChannelConfig, ChannelsConfig
from app.features.channels.exceptions import UnknownChannelKindError
from app.features.channels.registry import ChannelRegistry
from app.features.channels.telegram.config import TelegramChannelConfig


def test_channels_support_multiple_named_telegram_instances() -> None:
    config = ChannelsConfig(
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


def test_telegram_channel_requires_token() -> None:
    with pytest.raises(ValidationError):
        TelegramChannelConfig.model_validate({'kind': 'telegram'})


def test_channel_registry_rejects_unknown_kind() -> None:
    registry = ChannelRegistry()
    registry.register('telegram', TelegramChannelConfig)

    with pytest.raises(UnknownChannelKindError, match='discord'):
        registry.get('discord')


def test_base_channel_config_is_not_telegram_specific() -> None:
    channel = ChannelConfig(kind='custom', agent='scout')

    assert channel.kind == 'custom'
