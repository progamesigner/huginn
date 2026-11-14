import asyncio
import shutil
import subprocess
from functools import cache


@cache
def opus_encoder() -> str:
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise ValueError('Gemini speech requires ffmpeg with the libopus encoder')
    result = subprocess.run(  # noqa: S603
        [executable, '-hide_banner', '-encoders'],
        capture_output=True,
        timeout=5,
        check=True,
    )
    if b'libopus' not in result.stdout:
        raise ValueError('Gemini speech requires ffmpeg with the libopus encoder')
    return executable


async def encode_opus(data: bytes, *, sample_rate: int = 24000) -> bytes:
    process = await asyncio.create_subprocess_exec(
        opus_encoder(),
        '-hide_banner',
        '-loglevel',
        'error',
        '-nostdin',
        '-f',
        's16le',
        '-ar',
        str(sample_rate),
        '-ac',
        '1',
        '-i',
        'pipe:0',
        '-c:a',
        'libopus',
        '-b:a',
        '48k',
        '-f',
        'ogg',
        'pipe:1',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(30):
            output, _ = await process.communicate(data)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode != 0 or not output.startswith(b'OggS'):
        raise ValueError('ffmpeg could not encode the generated speech as ogg/opus')
    return output
