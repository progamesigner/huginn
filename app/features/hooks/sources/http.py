import asyncio
from collections.abc import Mapping

from httpx import URL
from pydantic import JsonValue

from app.features.hooks.config import HTTPHookConfig
from lib.http import HTTPClientContract

_LIMIT = 64 * 1024


async def read(
    http: HTTPClientContract,
    config: HTTPHookConfig,
    bound: Mapping[str, JsonValue],
) -> str:
    async with asyncio.timeout(config.timeout):
        async with (
            http.connect(headers=config.headers) as client,
            client.stream(
                config.method,
                URL(str(config.url)).copy_merge_params(
                    {key: str(value) for key, value in bound.items()},
                ),
                timeout=config.timeout,
            ) as response,
        ):
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk[: _LIMIT - len(body)])
                if len(body) == _LIMIT:
                    break
            return body.decode('utf-8', errors='replace')
