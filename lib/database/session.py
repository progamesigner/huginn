from sqlalchemy.ext.asyncio import AsyncSession


class ScopedSession:
    session: AsyncSession

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def dispose(self) -> None:
        await self.session.close()
