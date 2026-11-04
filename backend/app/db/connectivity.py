from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


class DatabaseConnectivityService:
    def __init__(self, engine: Engine):
        self._engine = engine

    def is_reachable(self) -> bool:
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
