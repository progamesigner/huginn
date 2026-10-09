from typing import Annotated, Literal

from pydantic import AliasChoices, Field

from app.core.schema import AppRequest, AppResponse


# NOTE: one shape for every way a file arrives; which fields are set says which way,
# and the service turns that into bytes or a refusal before any stream opens
class AttachmentInput(AppRequest):
    blob: Annotated[str | None, Field()] = None
    data: Annotated[str | None, Field()] = None
    mime: Annotated[str | None, Field()] = None
    name: Annotated[str | None, Field()] = None
    url: Annotated[str | None, Field()] = None


class ChatRequest(AppRequest):
    message: Annotated[str, Field()]
    attachments: Annotated[tuple[AttachmentInput, ...], Field(max_length=10)] = ()
    conversation: Annotated[str | None, Field()] = None
    agent: Annotated[str | None, Field()] = None
    max_completion_tokens: Annotated[int | None, Field(gt=0)] = None


class ImageURL(AppRequest):
    url: Annotated[str, Field()]


class InputAudio(AppRequest):
    data: Annotated[str, Field()]
    format: Annotated[str | None, Field()] = None


class InputFile(AppRequest):
    file_data: Annotated[str | None, Field()] = None
    file_id: Annotated[str | None, Field()] = None
    filename: Annotated[str | None, Field()] = None


class MediaSource(AppRequest):
    type: Annotated[str, Field()]
    data: Annotated[str | None, Field()] = None
    media_type: Annotated[str | None, Field()] = None
    url: Annotated[str | None, Field()] = None


class ContentPart(AppRequest):
    type: Annotated[str, Field()] = 'text'
    text: Annotated[str, Field()] = ''
    image_url: Annotated[ImageURL | None, Field()] = None
    input_audio: Annotated[InputAudio | None, Field()] = None
    file: Annotated[InputFile | None, Field()] = None
    source: Annotated[MediaSource | None, Field()] = None
    title: Annotated[str | None, Field()] = None

    def attachment(self) -> AttachmentInput | None:
        match self.type:
            case 'image_url' if self.image_url is not None:
                return _data_uri(self.image_url.url)
            case 'input_audio' if self.input_audio is not None:
                return AttachmentInput(data=self.input_audio.data)
            case 'file' if self.file is not None:
                return _file(self.file)
            case 'image' | 'document' if self.source is not None:
                return _source(self.source, self.title)
            case _:
                return None


def _data_uri(uri: str, name: str | None = None) -> AttachmentInput:
    if not uri.startswith('data:'):
        return AttachmentInput(url=uri)

    header, _, data = uri.partition(',')
    return AttachmentInput(data=data, mime=header[5:].split(';')[0] or None, name=name)


def _file(file: InputFile) -> AttachmentInput:
    if file.file_data is None:
        # NOTE: a file id names an upload in someone else's files api, not ours
        return AttachmentInput(url=file.file_id or '')

    if file.file_data.startswith('data:'):
        return _data_uri(file.file_data, file.filename)

    return AttachmentInput(data=file.file_data, name=file.filename)


def _source(source: MediaSource, title: str | None) -> AttachmentInput:
    if source.type == 'base64' and source.data is not None:
        return AttachmentInput(data=source.data, mime=source.media_type, name=title)

    return AttachmentInput(url=source.url or source.type)


class IncomingMessage(AppRequest):
    role: Annotated[str, Field()]
    content: Annotated[str | tuple[ContentPart, ...], Field()]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content

        return ''.join(part.text for part in self.content if part.type == 'text')

    def attachments(self) -> tuple[AttachmentInput, ...]:
        if isinstance(self.content, str):
            return ()

        return tuple(
            attachment
            for part in self.content
            if (attachment := part.attachment()) is not None
        )


class CompatibleRequest(AppRequest):
    model: Annotated[str, Field()] = ''
    messages: Annotated[tuple[IncomingMessage, ...], Field()] = ()
    stream: Annotated[bool, Field()] = False
    conversation: Annotated[str | None, Field()] = None
    user: Annotated[str | None, Field()] = None
    max_completion_tokens: Annotated[
        int | None,
        Field(
            gt=0,
            validation_alias=AliasChoices('max_completion_tokens', 'max_tokens'),
        ),
    ] = None

    def latest(self) -> str:
        message = self._latest()
        return '' if message is None else message.text()

    def latest_attachments(self) -> tuple[AttachmentInput, ...]:
        message = self._latest()
        return () if message is None else message.attachments()

    def _latest(self) -> IncomingMessage | None:
        return next(
            (message for message in reversed(self.messages) if message.role == 'user'),
            None,
        )

    def thread(self) -> str | None:
        return self.conversation or self.user


class CompletionMessage(AppResponse):
    role: Annotated[Literal['assistant'], Field()] = 'assistant'
    content: Annotated[str, Field()]


class CompletionChoice(AppResponse):
    index: Annotated[int, Field()] = 0
    message: Annotated[CompletionMessage, Field()]
    finish_reason: Annotated[str, Field()] = 'stop'


class Completion(AppResponse):
    id: Annotated[str, Field()]
    object: Annotated[Literal['chat.completion'], Field()] = 'chat.completion'
    created: Annotated[int, Field()]
    model: Annotated[str, Field()]
    choices: Annotated[tuple[CompletionChoice, ...], Field()]


class MessageContent(AppResponse):
    type: Annotated[Literal['text'], Field()] = 'text'
    text: Annotated[str, Field()]


class CompatibleMessage(AppResponse):
    id: Annotated[str, Field()]
    type: Annotated[Literal['message'], Field()] = 'message'
    role: Annotated[Literal['assistant'], Field()] = 'assistant'
    model: Annotated[str, Field()]
    content: Annotated[tuple[MessageContent, ...], Field()]
    stop_reason: Annotated[str, Field()] = 'end_turn'


class UploadedBlob(AppResponse):
    id: Annotated[str, Field()]
    kind: Annotated[str, Field()]
    mime: Annotated[str, Field()]
    size: Annotated[int, Field()]
    name: Annotated[str | None, Field()] = None
