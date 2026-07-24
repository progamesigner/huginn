from typing import Annotated, Literal

from pydantic import Field, SecretStr

from app.features.channels.config import ChannelConfig


class TelegramChannelConfig(ChannelConfig):
    kind: Annotated[Literal['telegram'], Field()]
    token: Annotated[SecretStr, Field()]
    secret_token: Annotated[SecretStr, Field()]
