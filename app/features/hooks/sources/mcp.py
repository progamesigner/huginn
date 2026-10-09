import json
from collections.abc import Mapping

from pydantic import JsonValue

from app.features.hooks.config import MCPHookConfig
from lib.mcp import MCPClientContract


def _content(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if data.get('isError'):
            raise ValueError('hook tool failed')
        for key in ('rendered', 'content'):
            if isinstance(data.get(key), str):
                return data[key]
    raise ValueError('hook response must contain text')


async def read(
    client: MCPClientContract,
    config: MCPHookConfig,
    bound: Mapping[str, JsonValue],
) -> str:
    return _content(
        await client.call_tool(
            config.server,
            config.tool,
            {**config.arguments, **bound},
        ),
    )
