from typing import override

from pydantic import BaseModel

from app.core.config.providers import PROVIDERS
from app.core.config.registry import HuginnConfig, SectionSpec
from app.core.providers.config import ConfigProvider
from app.features.agents.config import build_sections as agent_sections
from app.features.agents.contracts import (
    HookSource,
    ScopedCommandsServiceContract,
    SubagentServiceContract,
    ToolSource,
)
from app.features.agents.providers import AgentProvider
from app.features.assets.config import ASSETS, asset_settings
from app.features.assets.contracts import AssetsServiceContract
from app.features.assets.providers import AssetsProvider
from app.features.auth.config import build_sections as auth_sections
from app.features.auth.providers import AuthProvider
from app.features.blobs.config import BLOBS
from app.features.blobs.contracts import BlobReferenceSource, BlobServiceContract
from app.features.blobs.providers import BlobProvider
from app.features.briefs.providers import BriefsProvider
from app.features.briefs.tools import BriefToolSource
from app.features.channels.config import build_sections as channel_sections
from app.features.channels.handlers import CommandHandlers
from app.features.channels.providers import ChannelProvider
from app.features.channels.registry import (
    ChannelKind,
    ChannelRegistry,
    Command,
    CommandRegistry,
)
from app.features.channels.scoped_commands import ScopedCommands
from app.features.channels.telegram.config import TelegramChannelConfig
from app.features.conversations.config import COMPACTION
from app.features.conversations.contracts import ConversationServiceContract
from app.features.conversations.providers import ConversationProvider
from app.features.conversations.services import ConversationBlobReferences
from app.features.hooks.config import HOOKS
from app.features.hooks.providers import HooksProvider
from app.features.hooks.sources.configured import ConfiguredHookSource
from app.features.identities.contracts import (
    IdentityLinkingServiceContract,
    IdentityManagementServiceContract,
    IdentityResolutionServiceContract,
)
from app.features.identities.providers import IdentityProvider
from app.features.mcp.config import MCP_SERVERS
from app.features.mcp.providers import MCPProvider
from app.features.mcp.sources import MCPToolSource
from app.features.skills.config import SKILLS, effective_skills
from app.features.skills.providers import SkillsProvider
from app.features.skills.repositories import SkillRepositories
from app.features.skills.tools import SkillToolSource
from app.features.subagents.providers import SubagentsProvider
from app.features.subagents.tools import SubagentToolSource
from app.features.tools.artifacts import ArtifactWriter
from app.features.tools.config import build_sections as tool_sections
from app.features.tools.document.config import DocumentToolConfig
from app.features.tools.document.tool import ReadDocumentTool, build_read_document
from app.features.tools.images.config import ImageToolConfig
from app.features.tools.images.repositories import ImageRepository
from app.features.tools.images.tool import ImageTool, build_images
from app.features.tools.providers import ToolProvider
from app.features.tools.registry import ToolKind, ToolRegistry
from app.features.tools.sources import ConfiguredToolSource
from app.features.tools.speech.config import SpeechToolConfig
from app.features.tools.speech.repositories import SpeechRepository
from app.features.tools.speech.tool import SpeechTool, build_speech
from app.features.tools.websearch.config import WebSearchToolConfig
from app.features.tools.websearch.repositories import WebSearchRepository
from app.features.tools.websearch.tool import WebSearchTool, build_websearch
from app.features.tools.writing.config import WritingToolConfig
from app.features.tools.writing.tool import WriteDocumentTool, build_write_document
from lib.container import ContainerContract, ProviderContract
from lib.guid import GUIDContract
from lib.mcp import MCPClientContract
from lib.observability import Logger
from lib.temporal import ClockContract

# NOTE: every registry is filled here, at the composition root, so a feature never
# reaches into another's registry and every registry is built once and left alone


def _section(
    container: ContainerContract,
    spec: SectionSpec[BaseModel],
) -> None:
    container.contribute(SectionSpec, lambda _: spec)


class CoreContributions(ProviderContract):
    dependencies = (ConfigProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        _section(container, PROVIDERS)


class AgentContributions(ProviderContract):
    dependencies = (AgentProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        for spec in agent_sections():
            _section(container, spec)


class AuthContributions(ProviderContract):
    dependencies = (AuthProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        for spec in auth_sections():
            _section(container, spec)


class BlobContributions(ProviderContract):
    dependencies = (BlobProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        _section(container, BLOBS)


class AssetContributions(ProviderContract):
    dependencies = (AssetsProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        _section(container, ASSETS)


class ChannelContributions(ProviderContract):
    dependencies = (
        AssetContributions,
        ChannelProvider,
        ConversationProvider,
        IdentityProvider,
    )

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.contribute(
            ChannelKind,
            lambda _: ChannelKind('telegram', TelegramChannelConfig),
        )
        for spec in channel_sections(lambda: container.resolve(ChannelRegistry)):
            _section(container, spec)

        container.bind(
            ScopedCommandsServiceContract,
            factory=lambda c: ScopedCommands(
                c.resolve(HuginnConfig),
                asset_settings(c.resolve(HuginnConfig)),
                c.resolve(AssetsServiceContract),
                frozenset(c.resolve(CommandRegistry).names()),
                logger=c.resolve(Logger),
            ),
            shared=True,
        )

        handlers = CommandHandlers(
            lambda: container.resolve(IdentityResolutionServiceContract),
            lambda: container.resolve(IdentityLinkingServiceContract),
            lambda: container.resolve(ConversationServiceContract),
            lambda: container.resolve(CommandRegistry),
            lambda: container.resolve(IdentityManagementServiceContract),
        )
        for command in handlers.commands():
            container.contribute(Command, lambda _, command=command: command)


class ConversationContributions(ProviderContract):
    dependencies = (BlobProvider, ConversationProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        _section(container, COMPACTION)
        container.contribute(
            BlobReferenceSource,
            lambda c: ConversationBlobReferences(
                c.resolve(GUIDContract),
                c.resolve(ClockContract),
            ),
        )


class MCPContributions(ProviderContract):
    dependencies = (AgentProvider, MCPProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        _section(container, MCP_SERVERS)
        container.contribute(
            ToolSource,
            lambda c: MCPToolSource(
                c.resolve(MCPClientContract),
                logger=c.resolve(Logger),
            ),
        )


class SkillContributions(ProviderContract):
    dependencies = (AgentProvider, AssetContributions, SkillsProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        _section(container, SKILLS)
        container.contribute(
            ToolSource,
            lambda c: SkillToolSource(
                lambda name: c.resolve(SkillRepositories).get(name),
                effective_skills(c.resolve(HuginnConfig)),
                logger=c.resolve(Logger),
            ),
        )


class ToolContributions(ProviderContract):
    dependencies = (AgentProvider, BlobProvider, ToolProvider)

    @override
    def __call__(self, container: ContainerContract) -> None:
        for spec in tool_sections(lambda: container.resolve(ToolRegistry)):
            _section(container, spec)

        container.contribute(
            ToolKind,
            lambda c: ToolKind(
                WebSearchTool,
                WebSearchToolConfig,
                builder=lambda config, _scope: build_websearch(
                    c.resolve(WebSearchRepository),
                    config,
                ),
            ),
        )
        container.contribute(
            ToolKind,
            lambda c: ToolKind(
                ReadDocumentTool,
                DocumentToolConfig,
                builder=lambda config, scope: build_read_document(
                    c.resolve(BlobServiceContract),
                    config,
                    scope,
                ),
            ),
        )
        container.contribute(
            ToolKind,
            lambda c: ToolKind(
                ImageTool,
                ImageToolConfig,
                builder=lambda config, _scope: build_images(
                    c.resolve(ImageRepository),
                    c.resolve(ArtifactWriter),
                    config,
                ),
            ),
        )
        container.contribute(
            ToolKind,
            lambda c: ToolKind(
                SpeechTool,
                SpeechToolConfig,
                builder=lambda config, _scope: build_speech(
                    c.resolve(SpeechRepository),
                    c.resolve(ArtifactWriter),
                    config,
                ),
            ),
        )
        container.contribute(
            ToolKind,
            lambda c: ToolKind(
                WriteDocumentTool,
                WritingToolConfig,
                builder=lambda config, _scope: build_write_document(
                    c.resolve(ArtifactWriter),
                    config,
                ),
            ),
        )
        container.contribute(
            ToolSource,
            lambda c: ConfiguredToolSource(
                c.resolve(HuginnConfig),
                c.resolve(ToolRegistry),
                logger=c.resolve(Logger),
            ),
        )


class HookContributions(ProviderContract):
    dependencies = (HooksProvider,)

    @override
    def __call__(self, container: ContainerContract) -> None:
        _section(container, HOOKS)
        container.contribute(
            HookSource,
            lambda c: c.resolve(ConfiguredHookSource),
        )


class SubagentContributions(ProviderContract):
    dependencies = (SubagentsProvider, AssetContributions)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.contribute(
            ToolSource,
            lambda c: SubagentToolSource(
                c.resolve(AssetsServiceContract),
                asset_settings(c.resolve(HuginnConfig)),
                c.resolve(SubagentServiceContract),
                logger=c.resolve(Logger),
            ),
        )


class BriefContributions(ProviderContract):
    dependencies = (BriefsProvider, AssetContributions)

    @override
    def __call__(self, container: ContainerContract) -> None:
        container.contribute(
            ToolSource,
            lambda c: BriefToolSource(
                c.resolve(AssetsServiceContract),
                asset_settings(c.resolve(HuginnConfig)),
                c.resolve(MCPClientContract),
                logger=c.resolve(Logger),
            ),
        )
