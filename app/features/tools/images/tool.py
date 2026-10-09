from typing import Annotated, override

from pydantic import BaseModel, Field

from app.features.agents.contracts import ToolParams, ToolResult, TypedTool
from app.features.tools.artifacts import ArtifactWriter
from app.features.tools.exceptions import ToolError

from .config import ImageSize, ImageToolConfig
from .repositories import ImageRepository

_EXTENSIONS = {
    'image/gif': 'gif',
    'image/jpeg': 'jpg',
    'image/png': 'png',
    'image/webp': 'webp',
}


class ImageParams(ToolParams):
    prompt: Annotated[
        str,
        Field(description='What the image should show, described in detail'),
    ]
    size: Annotated[
        ImageSize | None,
        Field(description='Square, landscape or portrait; omit for the default'),
    ] = None


class ImageTool(TypedTool[ImageParams]):
    name = 'images'
    description = (
        'Generate an image from a description. The image is sent to the user'
        ' directly; you only get a note that it was made.'
    )
    params = ImageParams

    repository: ImageRepository
    writer: ArtifactWriter
    config: ImageToolConfig

    def __init__(
        self,
        repository: ImageRepository,
        writer: ArtifactWriter,
        config: ImageToolConfig,
    ) -> None:
        self.repository = repository
        self.writer = writer
        self.config = config

    @override
    async def run(self, params: ImageParams) -> ToolResult:
        size = params.size or self.config.size
        image = await self.repository.generate(self.config, params.prompt, size)
        artifact = await self.writer.write(
            image.mime,
            image.data,
            kind='image',
            name=f'image.{_EXTENSIONS.get(image.mime, "bin")}',
        )
        revised = (
            f' The generator read the request as: {image.revised_prompt}'
            if image.revised_prompt
            else ''
        )
        return ToolResult(
            text=f'Generated a {size} image and sent it to the user.{revised}',
            artifacts=(artifact,),
        )


def build_images(
    repository: ImageRepository,
    writer: ArtifactWriter,
    config: BaseModel,
) -> ImageTool:
    if not isinstance(config, ImageToolConfig):
        raise ToolError(
            f'images requires ImageToolConfig, got {type(config).__name__}',
        )

    return ImageTool(repository, writer, config)
