from __future__ import annotations

from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.config import create_database_url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine()


def create_engine() -> Engine:
    return sqlalchemy_create_engine(create_database_url(), pool_pre_ping=True)


def create_session_factory(engine: Engine | None = None) -> Any:
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)
