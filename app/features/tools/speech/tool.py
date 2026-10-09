from typing import Annotated, override

from pydantic import BaseModel, Field

from app.features.agents.contracts import ToolParams, ToolResult, TypedTool
from app.features.tools.artifacts import ArtifactWriter
from app.features.tools.exceptions import ToolError

from .config import SpeechFormat, SpeechToolConfig
from .repositories import SpeechRepository

_FORMATS: dict[SpeechFormat, tuple[str, str]] = {
    'aac': ('audio/aac', 'aac'),
    'flac': ('audio/flac', 'flac'),
    'mp3': ('audio/mpeg', 'mp3'),
    'opus': ('audio/ogg', 'ogg'),
    'wav': ('audio/wav', 'wav'),
}


class SpeechParams(ToolParams):
    text: Annotated[str, Field(min_length=1, description='The words to speak aloud')]
    voice: Annotated[
        str | None,
        Field(description='A voice name to use instead of the configured one'),
    ] = None


class SpeechTool(TypedTool[SpeechParams]):
    name = 'speech'
    description = (
        'Speak text aloud as a voice message sent to the user. Keep it to what is'
        ' worth hearing; one call makes one message.'
    )
    params = SpeechParams

    repository: SpeechRepository
    writer: ArtifactWriter
    config: SpeechToolConfig

    def __init__(
        self,
        repository: SpeechRepository,
        writer: ArtifactWriter,
        config: SpeechToolConfig,
    ) -> None:
        self.repository = repository
        self.writer = writer
        self.config = config

    # NOTE: too long is refused rather than split, because five voice notes in a row
    # is worse than a shorter one the model writes on a second try
    @override
    async def run(self, params: SpeechParams) -> ToolResult:
        if len(params.text) > self.config.max_chars:
            raise ToolError(
                f'the text is {len(params.text)} characters; speak at most'
                f' {self.config.max_chars} at a time, so shorten it',
            )

        data = await self.repository.synthesize(
            self.config,
            params.text,
            params.voice or self.config.voice,
        )
        mime, extension = _FORMATS[self.config.format]
        artifact = await self.writer.write(
            mime,
            data,
            kind='audio',
            name=f'speech.{extension}',
        )
        return ToolResult(
            text=f'Spoke {len(params.text)} characters as a voice message.',
            artifacts=(artifact,),
        )


def build_speech(
    repository: SpeechRepository,
    writer: ArtifactWriter,
    config: BaseModel,
) -> SpeechTool:
    if not isinstance(config, SpeechToolConfig):
        raise ToolError(
            f'speech requires SpeechToolConfig, got {type(config).__name__}',
        )

    return SpeechTool(repository, writer, config)
