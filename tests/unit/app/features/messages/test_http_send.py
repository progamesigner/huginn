import json
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.contracts.channels import ReplyArtifact
from app.core.config.registry import HuginnConfig
from app.features.agents.config import (
    AGENTS,
    DEFAULTS,
    AgentConfig,
    AgentConfigList,
    DefaultsConfig,
)
from app.features.auth.config import API_KEYS, ApiKeyConfig, ApiKeyConfigList
from app.features.blobs.config import BLOBS, BlobsConfig
from app.features.channels.config import TRANSCRIPTION, TranscriptionConfig
from app.features.messages.handlers import disposition
from tests.helpers.blobs import MemoryBlobs

from .test_native_chat import (
    _CREDENTIAL,
    StreamingTurns,
    api_key,
    chat_client,
    events,
    make_config,
)

_CHART = b'\x89PNG chart bytes'
_OTHER = 'other-key'


async def produced(blobs: MemoryBlobs, name: str = 'chart.png') -> ReplyArtifact:
    blob = await blobs.save('image/png', _CHART, kind='image', filename=name)
    return ReplyArtifact(
        blob=str(blob.id),
        kind='image',
        mime='image/png',
        size=len(_CHART),
        data=_CHART,
        name=name,
    )


def two_keys() -> HuginnConfig:
    return HuginnConfig(
        sections={
            AGENTS.key: AgentConfigList(
                {'scout': AgentConfig(description='research', instructions='scout')},
            ),
            API_KEYS.key: ApiKeyConfigList(
                {
                    'web': api_key(agent='scout'),
                    'other': ApiKeyConfig(
                        token=SecretStr(_OTHER),
                        agents=('scout',),
                        agent='scout',
                    ),
                },
            ),
            DEFAULTS.key: DefaultsConfig(model='gpt-5-mini'),
            BLOBS.key: BlobsConfig(),
            TRANSCRIPTION.key: TranscriptionConfig(),
        },
    )


def bearer(token: str = _CREDENTIAL) -> dict[str, str]:
    return {'Authorization': f'Bearer {token}'}


async def test_a_produced_file_is_announced_without_its_bytes() -> None:
    blobs = MemoryBlobs()
    artifact = await produced(blobs)
    turns = StreamingTurns('here', artifacts=(artifact,))

    async with chat_client(
        make_config(api_key(agent='scout')),
        turns,
        blobs=blobs,
    ) as client:
        response = await client.post(
            '/v1/chat',
            headers=bearer(),
            json={'message': 'x'},
        )

    named = events(response.text)
    assert [name for name, _ in named] == [
        'opened',
        'routed',
        'delta',
        'attachment',
        'done',
    ]
    assert json.loads(named[3][1]) == {
        'blob': artifact.blob,
        'kind': 'image',
        'mime': 'image/png',
        'size': len(_CHART),
        'url': f'/v1/blobs/{artifact.blob}',
        'name': 'chart.png',
    }


@pytest.mark.parametrize(
    ('path', 'stream'),
    [
        ('/v1/chat', True),
        ('/openai/v1/chat/completions', True),
        ('/openai/v1/chat/completions', False),
        ('/anthropic/v1/messages', True),
        ('/anthropic/v1/messages', False),
    ],
)
async def test_a_failed_grant_replaces_the_link_with_a_notice(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    stream: bool,
) -> None:
    blobs = MemoryBlobs()
    artifact = await produced(blobs)
    grant = AsyncMock(side_effect=RuntimeError('grant failed'))
    monkeypatch.setattr(blobs, 'grant', grant)
    turns = StreamingTurns('here', artifacts=(artifact,))
    payload = (
        {'message': 'x'}
        if path == '/v1/chat'
        else {'messages': [{'role': 'user', 'content': 'x'}], 'stream': stream}
    )

    async with chat_client(
        make_config(api_key(agent='scout')),
        turns,
        blobs=blobs,
    ) as client:
        response = await client.post(path, headers=bearer(), json=payload)

    assert response.status_code == 200
    grant.assert_awaited_once_with(
        UUID(artifact.blob),
        'web',
        filename='chart.png',
    )
    assert artifact.blob not in response.text
    assert '/v1/blobs/' not in response.text
    notice = '\n\nThe attachment could not be made available.'
    if path == '/v1/chat':
        named = events(response.text)
        assert [name for name, _ in named] == [
            'opened',
            'routed',
            'delta',
            'delta',
            'done',
        ]
        assert json.loads(named[3][1]) == {'text': notice}
    elif path == '/openai/v1/chat/completions':
        if stream:
            chunks = [
                json.loads(data)
                for _, data in events(response.text)
                if data and data != '[DONE]'
            ]
            text = ''.join(
                chunk['choices'][0]['delta'].get('content', '') for chunk in chunks
            )
            assert events(response.text)[-1][1] == '[DONE]'
        else:
            text = response.json()['choices'][0]['message']['content']
        assert text == 'here' + notice
    else:
        if stream:
            named = events(response.text)
            text = ''.join(
                json.loads(data)['delta']['text']
                for name, data in named
                if name == 'content_block_delta'
            )
            assert named[-1][0] == 'message_stop'
        else:
            text = response.json()['content'][0]['text']
        assert text == 'here' + notice


async def test_the_announced_url_serves_the_bytes_to_the_same_key() -> None:
    blobs = MemoryBlobs()
    artifact = await produced(blobs)
    turns = StreamingTurns('here', artifacts=(artifact,))

    async with chat_client(two_keys(), turns, blobs=blobs) as client:
        await client.post('/v1/chat', headers=bearer(), json={'message': 'x'})
        fetched = await client.get(f'/v1/blobs/{artifact.blob}', headers=bearer())
        stranger = await client.get(
            f'/v1/blobs/{artifact.blob}',
            headers=bearer(_OTHER),
        )

    assert fetched.status_code == 200
    assert fetched.content == _CHART
    assert fetched.headers['content-type'] == 'image/png'
    assert fetched.headers['content-disposition'] == (
        'attachment; filename="chart.png"; filename*=UTF-8\'\'chart.png'
    )
    # NOTE: 404 rather than 403, so a guessed id never learns that it exists
    assert stranger.status_code == 404
    assert stranger.json()['error'] == 'BLOB_NOT_FOUND'


async def test_a_blob_nobody_granted_or_a_malformed_id_is_not_found() -> None:
    blobs = MemoryBlobs()
    artifact = await produced(blobs)

    async with chat_client(
        make_config(api_key(agent='scout')),
        StreamingTurns(),
        blobs=blobs,
    ) as client:
        ungranted = await client.get(f'/v1/blobs/{artifact.blob}', headers=bearer())
        malformed = await client.get('/v1/blobs/not-an-id', headers=bearer())
        anonymous = await client.get(f'/v1/blobs/{artifact.blob}')

    assert ungranted.status_code == 404
    assert malformed.status_code == 404
    assert anonymous.status_code == 401


def test_a_hostile_file_name_cannot_break_out_of_its_header() -> None:
    header = disposition('../../etc/pa"ss\r\nX-Evil: 1.png')

    assert '\r' not in header
    assert '\n' not in header
    assert header.startswith('attachment; filename="')
    plain = header.split('filename="')[1].split('"')[0]
    assert '/' not in plain
    assert '"' not in plain


def test_a_nameless_or_unicode_file_name_still_downloads() -> None:
    assert disposition(None) == (
        'attachment; filename="download"; filename*=UTF-8\'\'download'
    )
    assert disposition('報表.xlsx') == (
        'attachment; filename="download.xlsx"; filename*=UTF-8\'\'%E5%A0%B1%E8%A1%A8.xlsx'
    )


async def test_an_openai_stream_links_the_file_after_the_answer() -> None:
    blobs = MemoryBlobs()
    artifact = await produced(blobs, name='chart [v2].png')
    turns = StreamingTurns('here', artifacts=(artifact,))

    async with chat_client(
        make_config(api_key(agent='scout')),
        turns,
        blobs=blobs,
    ) as client:
        streamed = await client.post(
            '/openai/v1/chat/completions',
            headers=bearer(),
            json={'messages': [{'role': 'user', 'content': 'x'}], 'stream': True},
        )
        whole = await client.post(
            '/openai/v1/chat/completions',
            headers=bearer(),
            json={'messages': [{'role': 'user', 'content': 'x'}]},
        )

    link = f'\n\n[chart (v2).png](/v1/blobs/{artifact.blob})'
    chunks = [
        json.loads(data)
        for _, data in events(streamed.text)
        if data and data != '[DONE]'
    ]
    text = ''.join(chunk['choices'][0]['delta'].get('content', '') for chunk in chunks)
    assert text == 'here' + link
    assert 'attachment' not in streamed.text.split('data:')[0]
    assert whole.json()['choices'][0]['message']['content'] == 'here' + link


async def test_an_anthropic_answer_links_the_file_after_the_answer() -> None:
    blobs = MemoryBlobs()
    artifact = await produced(blobs)
    turns = StreamingTurns('here', artifacts=(artifact,))

    async with chat_client(
        make_config(api_key(agent='scout')),
        turns,
        blobs=blobs,
    ) as client:
        streamed = await client.post(
            '/anthropic/v1/messages',
            headers=bearer(),
            json={'messages': [{'role': 'user', 'content': 'x'}], 'stream': True},
        )

    named = events(streamed.text)
    assert 'attachment' not in [name for name, _ in named]
    deltas = [
        json.loads(data)['delta']['text']
        for name, data in named
        if name == 'content_block_delta'
    ]
    assert ''.join(deltas) == f'here\n\n[chart.png](/v1/blobs/{artifact.blob})'
