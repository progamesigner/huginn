from base64 import b64encode
from datetime import timedelta

import httpx2
import pytest
from pydantic import HttpUrl, SecretStr

from app.exts.telegram.clients import TelegramClient
from app.exts.telegram.gateways import TelegramReply
from app.features.agents.services import AgentRunner
from app.features.agents.tooling import ToolSet
from app.features.tools.artifacts import ArtifactWriter
from app.features.tools.exceptions import ToolError
from app.features.tools.images.config import ImageToolConfig
from app.features.tools.images.repositories import ImageRepository
from app.features.tools.images.tool import ImageParams, ImageTool, build_images
from app.features.tools.websearch.config import WebSearchToolConfig
from lib.llm import FinishChunk, TextChunk, ToolCallChunk, ToolCallSegment
from lib.temporal import ClockManager
from tests.helpers.blobs import MemoryBlobs
from tests.helpers.clock import FakeClock
from tests.helpers.http import MockHTTP, Route
from tests.helpers.logging import make_logger
from tests.unit.app.features.agents.test_agent_runner import FakeLLM, make_turn

_VENDOR = 'https://images.example'
_TELEGRAM = 'https://api.telegram.org/bottelegram-token'
_PNG = b'\x89PNG\r\n\x1a\n image'
_TTL = timedelta(days=30)


def config(**overrides: object) -> ImageToolConfig:
    return ImageToolConfig.model_validate(
        {
            'kind': 'images',
            'model': 'gpt-image-1',
            'api_key': 'image-key',
            'base_url': _VENDOR,
            **overrides,
        },
    )


def make_tool(http: MockHTTP, blobs: MemoryBlobs) -> ImageTool:
    return build_images(
        ImageRepository(http.manager()),
        ArtifactWriter(blobs, _TTL),
        config(),
    )


def vendor(http: MockHTTP, data: bytes = _PNG, status: int = 200) -> Route:
    body = (
        {'data': [{'b64_json': b64encode(data).decode()}]}
        if status == 200
        else {'error': {'message': 'no'}}
    )
    return http.post(f'{_VENDOR}/v1/images/generations').mock(
        return_value=httpx2.Response(status, json=body),
    )


def test_params_offer_only_the_sizes_the_vendor_draws() -> None:
    schema = ImageParams.model_json_schema()

    assert schema['required'] == ['prompt']
    assert ImageParams.model_validate({'prompt': 'x'}).size is None
    with pytest.raises(ValueError, match='size'):
        ImageParams.model_validate({'prompt': 'x', 'size': '9x9'})


def test_build_images_refuses_another_tools_config(http: MockHTTP) -> None:
    other = WebSearchToolConfig.model_validate({'kind': 'websearch', 'api_key': 'k'})

    with pytest.raises(ToolError, match='ImageToolConfig'):
        build_images(
            ImageRepository(http.manager()),
            ArtifactWriter(MemoryBlobs(), _TTL),
            other,
        )


async def test_a_generated_image_is_stored_and_handed_over_beside_a_note(
    http: MockHTTP,
) -> None:
    route = vendor(http)
    blobs = MemoryBlobs()

    result = await make_tool(http, blobs).run(
        ImageParams(prompt='a duck', size='1536x1024'),
    )

    assert result.text == 'Generated a 1536x1024 image and sent it to the user.'
    (artifact,) = result.artifacts
    assert (artifact.kind, artifact.mime, artifact.name) == (
        'image',
        'image/png',
        'image.png',
    )
    assert artifact.data == _PNG
    stored = next(iter(blobs.blobs.values()))
    assert str(stored.id) == artifact.blob
    assert blobs.ttls[stored.id] == _TTL
    assert b'1536x1024' in route.calls.last.request.content


async def test_the_configured_size_applies_when_the_model_names_none(
    http: MockHTTP,
) -> None:
    route = vendor(http)

    await make_tool(http, MemoryBlobs()).run(ImageParams(prompt='a duck'))

    assert b'"size":"1024x1024"' in route.calls.last.request.content.replace(b' ', b'')


async def test_a_refusal_reaches_the_model_as_a_tool_error(http: MockHTTP) -> None:
    vendor(http, status=400)
    tools = ToolSet(
        {'illustrator': make_tool(http, MemoryBlobs())},
        logger=make_logger(),
    )

    result, artifacts = await tools.execute(
        ToolCallSegment(id='call-1', name='illustrator', arguments='{"prompt":"x"}'),
    )

    assert result.error is True
    assert 'refused the request (400): no' in result.content
    assert artifacts == ()


async def test_bytes_that_are_not_an_image_are_refused(http: MockHTTP) -> None:
    vendor(http, data=b'%PDF-1.7 not an image')

    with pytest.raises(ToolError, match='not an image'):
        await make_tool(http, MemoryBlobs()).run(ImageParams(prompt='a duck'))


async def test_an_image_asked_for_on_telegram_arrives_as_a_photo(
    http: MockHTTP,
) -> None:
    vendor(http)
    http.post(f'{_TELEGRAM}/sendMessage').mock(
        return_value=httpx2.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'message_id': 1,
                    'date': 1_721_600_000,
                    'chat': {'id': 5, 'type': 'private'},
                },
            },
        ),
    )
    photo = http.post(f'{_TELEGRAM}/sendPhoto').mock(
        return_value=httpx2.Response(
            200,
            json={
                'ok': True,
                'result': {
                    'message_id': 2,
                    'date': 1_721_600_000,
                    'chat': {'id': 5, 'type': 'private'},
                },
            },
        ),
    )
    llm = FakeLLM(
        [
            [
                ToolCallChunk(
                    index=0,
                    id='call_1',
                    name='illustrator',
                    arguments='{"prompt": "a duck"}',
                ),
                FinishChunk(reason='tool_calls'),
            ],
            [TextChunk(text='There you go.'), FinishChunk(reason='stop')],
        ],
    )
    tools = ToolSet(
        {'illustrator': make_tool(http, MemoryBlobs())},
        logger=make_logger(),
    )
    reply = TelegramReply(
        TelegramClient(http=http.manager(), bot_token=SecretStr('telegram-token')),
        5,
        FakeClock(),
        logger=make_logger(),
    )

    await AgentRunner(llm, tools, ClockManager(), logger=make_logger()).run(
        make_turn(),
        reply,
    )
    await reply.finish()

    assert photo.called
    assert _PNG in photo.calls.last.request.content


@pytest.mark.parametrize('max_bytes', [0, -1])
def test_image_config_requires_a_positive_size_limit(max_bytes: int) -> None:
    with pytest.raises(ValueError, match='max_bytes'):
        config(max_bytes=max_bytes)


@pytest.mark.parametrize('linked', [False, True])
async def test_an_oversized_image_is_not_stored(http: MockHTTP, linked: bool) -> None:
    if linked:
        http.post(f'{_VENDOR}/v1/images/generations').mock(
            return_value=httpx2.Response(
                200,
                json={'data': [{'url': 'https://storage.example/duck.png'}]},
            ),
        )
        http.get('https://storage.example/duck.png').mock(
            return_value=httpx2.Response(200, content=_PNG),
        )
    else:
        vendor(http)
    blobs = MemoryBlobs()
    tool = build_images(
        ImageRepository(http.manager()),
        ArtifactWriter(blobs, _TTL),
        config(max_bytes=len(_PNG) - 1),
    )

    with pytest.raises(ToolError, match='size limit'):
        await tool.run(ImageParams(prompt='a duck'))

    assert blobs.blobs == {}


def test_gemini_images_preserve_an_explicit_endpoint_and_warn_on_quality() -> None:
    import pytest

    from app.features.tools.images.config import ImageToolConfig

    with pytest.warns(UserWarning, match='ignore quality'):
        config = ImageToolConfig(
            kind='images',
            vendor='gemini',
            model='test',
            api_key=SecretStr('test'),
            base_url=HttpUrl('https://gateway.example'),
            quality='high',
        )
    assert str(config.base_url) == 'https://gateway.example/'
