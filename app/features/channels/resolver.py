import hmac
from typing import cast, override

from pydantic import SecretStr

from app.contracts.channels import ChannelResolverContract
from app.core.config.registry import HuginnConfig

from .config import CHANNELS


class ChannelSecretResolver(ChannelResolverContract):
    config: HuginnConfig

    def __init__(self, config: HuginnConfig) -> None:
        self.config = config

    @override
    def resolve(self, kind: str, credential: str) -> str | None:
        for name, channel in self.config.section(CHANNELS).root.items():
            secret = cast(SecretStr | None, getattr(channel, 'secret_token', None))
            if (
                channel.kind == kind
                and secret is not None
                and hmac.compare_digest(secret.get_secret_value(), credential)
            ):
                return name

        return None
