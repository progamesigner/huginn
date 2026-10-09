from typing import override

from app.contracts.channels import (
    ChannelReplyContract,
    ReplyArtifact,
    ToolPhase,
    TurnEnding,
)


class RecordingReply(ChannelReplyContract):
    abandoned: bool
    artifacts: list[ReplyArtifact]
    endings: list[TurnEnding]
    finished: bool
    fragments: list[str]
    routed: list[str]
    tools: list[tuple[str, ToolPhase]]

    def __init__(self) -> None:
        self.abandoned = False
        self.artifacts = []
        self.endings = []
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
    async def on_artifact(self, artifact: ReplyArtifact) -> None:
        self.artifacts.append(artifact)

    @override
    async def on_routed(self, agent: str) -> None:
        self.routed.append(agent)

    @override
    async def on_text(self, fragment: str) -> None:
        self.fragments.append(fragment)

    @override
    async def on_tool(self, name: str, phase: ToolPhase) -> None:
        self.tools.append((name, phase))

    @override
    async def unfinished(self, ending: TurnEnding) -> None:
        self.endings.append(ending)
