from typing import override

from app.contracts.channels import ChannelReplyContract, ToolPhase


class RecordingReply(ChannelReplyContract):
    abandoned: bool
    finished: bool
    fragments: list[str]
    routed: list[str]
    tools: list[tuple[str, ToolPhase]]

    def __init__(self) -> None:
        self.abandoned = False
        self.finished = False
        self.fragments = []
        self.routed = []
        self.tools = []

    @override
    async def abandon(self) -> None:
        self.abandoned = True

    @override
    async def finish(self) -> None:
        self.finished = True

    @override
    async def on_routed(self, agent: str) -> None:
        self.routed.append(agent)

    @override
    async def on_text(self, fragment: str) -> None:
        self.fragments.append(fragment)

    @override
    async def on_tool(self, name: str, phase: ToolPhase) -> None:
        self.tools.append((name, phase))
