from __future__ import annotations

from backend.app.core.config import Settings


def build_database_url(
    *,
    host: str,
    port: int,
    database: str | None,
    user: str | None,
    password: str | None,
) -> str:
    if not database:
        raise ValueError("POSTGRES_DB is required to construct the database URL.")
    if not user:
        raise ValueError("POSTGRES_USER is required to construct the database URL.")

    credentials = f"{user}:{password or ''}"
    return f"postgresql+psycopg://{credentials}@{host}:{port}/{database}"


def create_database_url(
    *,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> str:
    configuration = Settings()
    return build_database_url(
        host=host or configuration.POSTGRES_HOST,
        port=port or configuration.POSTGRES_PORT,
        database=database if database is not None else configuration.POSTGRES_DB,
        user=user if user is not None else configuration.POSTGRES_USER,
        password=password if password is not None else configuration.POSTGRES_PASSWORD,
    )
