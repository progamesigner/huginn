from typing import Annotated

from app.core.dependencies import Provide
from lib.redis import RedisContract
from lib.router import Depends, Router

router = Router()


@router.get('/healthz')
async def healthz() -> dict[str, str]:
    return {'status': 'ok'}


@router.get('/readyz')
async def readyz(
    redis: Annotated[RedisContract, Depends(Provide(RedisContract))],
) -> dict[str, str]:
    async with redis.connect() as client:
        await client.ping()
    return {'status': 'ok'}
