from typing import Annotated, Literal, cast

import pytest
from pydantic import Field, ValidationError

from app.features.channels.config import (
    ChannelConfig,
    ChannelsConfig,
    _build_channels_config,
)
from app.features.channels.registry import ChannelRegistry
from app.features.channels.telegram.config import TelegramChannelConfig


class DiscordChannelConfig(ChannelConfig):
    kind: Annotated[Literal['discord'], Field()]
    webhook_url: Annotated[str, Field()]


def test_registered_channel_kind_extends_channels_schema() -> None:
    registry = ChannelRegistry()
    registry.register('telegram', TelegramChannelConfig)
    registry.register('discord', DiscordChannelConfig)
    model = _build_channels_config(registry)

    channels = cast(
        ChannelsConfig,
        model.model_validate(
            {
                'alerts': {
                    'kind': 'discord',
                    'webhook_url': 'https://discord.example/webhook',
                    'agent': 'operator',
                },
            },
        ),
    )

    assert isinstance(channels.root['alerts'], DiscordChannelConfig)


def test_unregistered_channel_kind_fails_schema_validation() -> None:
    registry = ChannelRegistry()
    registry.register('telegram', TelegramChannelConfig)
    model = _build_channels_config(registry)

    with pytest.raises(ValidationError, match='discord'):
        model.model_validate({'alerts': {'kind': 'discord'}})
