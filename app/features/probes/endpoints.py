from typing import Annotated

from app.core.dependencies import Provide
from lib.database import DatabaseContract
from lib.mcp import MCPClientContract
from lib.redis import RedisContract
from lib.router import Depends, Router

router = Router()


@router.get('/healthz')
async def healthz() -> dict[str, str]:
    return {'status': 'ok'}


@router.get('/readyz')
async def readyz(
    database: Annotated[DatabaseContract, Depends(Provide(DatabaseContract))],
    mcp: Annotated[MCPClientContract, Depends(Provide(MCPClientContract))],
    redis: Annotated[RedisContract, Depends(Provide(RedisContract))],
) -> dict[str, str]:
    await database.ping()

    async with redis.connect() as client:
        await client.ping()

    unavailable = mcp.unavailable()
    return {
        'status': 'ok',
        'mcp': f'unavailable: {", ".join(unavailable)}' if unavailable else 'ok',
    }
