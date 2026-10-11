import asyncio
import os
from collections.abc import Mapping

from pydantic import JsonValue

from app.features.hooks.config import ShellHookConfig

_LIMIT = 64 * 1024


async def read(config: ShellHookConfig, bound: Mapping[str, JsonValue]) -> str:
    process = await asyncio.create_subprocess_exec(
        *config.command,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if key in ('PATH', 'HOME')
                or key.startswith('HUGINN_HOOK_')
                or key in config.inherit_env
            },
            **{
                f'HUGINN_HOOK_{key.upper()}': str(value) for key, value in bound.items()
            },
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    if process.stdout is None:
        raise RuntimeError('hook stdout pipe was not created')
    body = bytearray()
    try:
        async with asyncio.timeout(config.timeout):
            while chunk := await process.stdout.read(8192):
                body.extend(chunk[: max(0, _LIMIT - len(body))])
            if await process.wait() != 0:
                raise ValueError('hook command failed')
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    return body.decode('utf-8', errors='replace')
