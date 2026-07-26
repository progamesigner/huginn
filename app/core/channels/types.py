from dataclasses import dataclass


@dataclass(frozen=True)
class InboundMessage:
    channel: str
    chat: str
    event: str
    raw: bytes
