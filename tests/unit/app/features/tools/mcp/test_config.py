import pytest
from pydantic import HttpUrl, SecretStr, ValidationError

from app.features.tools.mcp.config import McpServerConfig, McpServersConfig


def test_mcp_servers_config_holds_named_declarations() -> None:
    servers = McpServersConfig(
        {
            'memory': McpServerConfig(
                transport='http',
                url=HttpUrl('https://memory.example.internal/mcp'),
                auth=SecretStr('token'),
            ),
        }
    )

    assert servers.root['memory'].transport == 'http'


def test_mcp_server_config_requires_transport_and_url() -> None:
    with pytest.raises(ValidationError, match='transport'):
        McpServerConfig.model_validate({'url': 'https://memory.example.internal/mcp'})

    with pytest.raises(ValidationError, match='url'):
        McpServerConfig.model_validate({'transport': 'http'})


def test_mcp_server_config_rejects_unknown_transport() -> None:
    with pytest.raises(ValidationError):
        McpServerConfig.model_validate(
            {'transport': 'stdio', 'url': 'https://memory.example.internal/mcp'}
        )
