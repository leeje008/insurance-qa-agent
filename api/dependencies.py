"""api.dependencies - FastAPI 의존성 주입."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from api.db.database import async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """DB 세션 의존성."""
    async with async_session() as session:
        yield session
