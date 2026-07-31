from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr

from app.features.channels.config import ChannelConfig


class TelegramChannelConfig(ChannelConfig):
    kind: Annotated[Literal['telegram'], Field()]
    token: Annotated[SecretStr, Field()]
    secret_token: Annotated[SecretStr, Field()]
    webhook_base_url: Annotated[HttpUrl | None, Field()] = None
