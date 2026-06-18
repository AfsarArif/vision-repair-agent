from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import create_engine

from repair_agent.config import settings

# Async engine for FastAPI + LangGraph
async_engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for Alembic migrations + pgvector (required by langchain-postgres)
sync_engine = create_engine(settings.SYNC_DATABASE_URL, echo=False, pool_size=5)
