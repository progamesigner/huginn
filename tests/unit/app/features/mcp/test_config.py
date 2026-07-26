import pytest
from pydantic import HttpUrl, SecretStr, ValidationError

from app.features.mcp.config import MCPServerConfig, MCPServerConfigList


def test_mcp_servers_config_holds_named_declarations() -> None:
    servers = MCPServerConfigList(
        {
            'memory': MCPServerConfig(
                transport='http',
                url=HttpUrl('https://memory.example.internal/mcp'),
                auth=SecretStr('token'),
            ),
        },
    )

    assert servers.root['memory'].transport == 'http'


def test_mcp_server_config_requires_transport_and_url() -> None:
    with pytest.raises(ValidationError, match='transport'):
        MCPServerConfig.model_validate({'url': 'https://memory.example.internal/mcp'})

    with pytest.raises(ValidationError, match='url'):
        MCPServerConfig.model_validate({'transport': 'http'})


def test_mcp_server_config_rejects_unknown_transport() -> None:
    with pytest.raises(ValidationError):
        MCPServerConfig.model_validate(
            {'transport': 'stdio', 'url': 'https://memory.example.internal/mcp'},
        )
