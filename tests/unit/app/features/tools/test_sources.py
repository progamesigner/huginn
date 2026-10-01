import json
from datetime import timedelta
from typing import Annotated, override
from uuid import UUID, uuid4

from pydantic import Field

from app.core.config.registry import HuginnConfig
from app.features.agents.config import AgentConfig, ToolRef
from app.features.agents.types import ToolOutput, ToolParams, TurnScope, TypedTool
from app.features.blobs.contracts import BlobStoreContract
from app.features.blobs.types import Blob, BlobKind
from app.features.tools.config import ToolConfig, ToolConfigList
from app.features.tools.document.config import DocumentToolConfig
from app.features.tools.document.tool import ReadDocumentTool, build_read_document
from app.features.tools.registry import ToolKind, ToolRegistry
from app.features.tools.sources import ConfiguredToolSource


class EchoParams(ToolParams):
    message: Annotated[str, Field()]
    user: Annotated[str, Field()]


class EchoTool(TypedTool[EchoParams]):
    name = 'echo'
    description = 'Echo a message'
    params = EchoParams

    @override
    async def run(self, params: EchoParams) -> ToolOutput:
        return f'{params.user}:{params.message}'


def _source() -> ConfiguredToolSource:
    registry = ToolRegistry(
        [
            ToolKind(EchoTool),
        ],
    )
    config = HuginnConfig(
        sections={'tools': ToolConfigList({'echo': ToolConfig(kind='echo')})},
    )
    return ConfiguredToolSource(config, registry)


async def test_configured_tool_source_binds_and_hides_turn_value() -> None:
    agent = AgentConfig(
        description='test agent',
        tools=(ToolRef(name='echo', bind={'user': 'handle'}),),
    )
    scope = TurnScope(
        actor='telegram:42',
        agent='scout',
        channel='telegram',
        chat='42',
        handle='alice',
    )

    harvest = await _source().collect(agent, scope)
    tool = harvest.tools[0][1]
    schema = tool.schema()

    assert isinstance(schema, dict)
    properties = schema['properties']
    assert isinstance(properties, dict)
    assert 'user' not in properties
    assert await tool.invoke(json.dumps({'message': 'hello', 'user': 'mallory'})) == (
        'alice:hello'
    )


async def test_configured_tool_source_omits_ref_with_missing_binding() -> None:
    agent = AgentConfig(
        description='test agent',
        tools=(ToolRef(name='echo', bind={'user': 'handle'}),),
    )
    scope = TurnScope(
        actor='telegram:42',
        agent='scout',
        channel='telegram',
        chat='42',
    )

    harvest = await _source().collect(agent, scope)

    assert harvest.tools == ()


class DocumentBlobs(BlobStoreContract):
    def __init__(self, blob: Blob) -> None:
        self.blob = blob

    @override
    async def load(self, id: UUID) -> Blob | None:
        return self.blob if id == self.blob.id else None

    @override
    async def head(self, id: UUID) -> Blob | None:
        return None

    @override
    async def store(
        self,
        mime: str,
        data: bytes,
        *,
        kind: BlobKind,
        filename: str | None = None,
        ttl: timedelta | None = None,
    ) -> Blob:
        raise NotImplementedError


async def test_document_source_applies_turn_allow_list() -> None:
    blob_id = uuid4()
    blobs = DocumentBlobs(
        Blob(blob_id, 'hash', 'text/markdown', 5, 'document', data=b'hello'),
    )
    registry = ToolRegistry(
        [
            ToolKind(
                ReadDocumentTool,
                DocumentToolConfig,
                builder=lambda config, scope: build_read_document(blobs, config, scope),
            ),
        ],
    )
    config = HuginnConfig(
        sections={
            'tools': ToolConfigList({'read': DocumentToolConfig(kind='read_document')}),
        },
    )
    source = ConfiguredToolSource(config, registry)
    agent = AgentConfig(description='test', tools=(ToolRef(name='read'),))
    scope = TurnScope(
        actor='user:1',
        agent='test',
        channel='web',
        chat='1',
        document_blobs=frozenset({str(blob_id)}),
    )

    harvest = await source.collect(agent, scope)
    tool = harvest.tools[0][1]
    assert (await tool.invoke(json.dumps({'blob': str(blob_id)}))).endswith('\nhello')
