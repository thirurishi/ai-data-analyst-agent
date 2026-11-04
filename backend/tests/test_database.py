from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from backend.app.db.config import build_database_url, create_database_url
from backend.app.db.connectivity import DatabaseConnectivityService
from backend.app.db.engine import create_engine, get_engine
from backend.app.db.session import session_scope


def test_database_url_uses_postgres_environment_values():
    url = build_database_url(
        host="db.example.test",
        port=5433,
        database="analytics",
        user="analyst",
        password="secret",
    )

    assert url == "postgresql+psycopg://analyst:secret@db.example.test:5433/analytics"


def test_create_database_url_rejects_missing_configuration():
    with pytest.raises(ValueError, match="POSTGRES_DB"):
        create_database_url(
            host="localhost",
            port=5432,
            database=None,
            user="analyst",
            password="secret",
        )


def test_engine_is_created_lazily_and_is_reusable(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "analytics")
    monkeypatch.setenv("POSTGRES_USER", "analyst")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    first = get_engine()
    second = get_engine()

    assert first is second
    assert first.url.drivername == "postgresql+psycopg"


def test_session_scope_closes_session(monkeypatch):
    class TrackingSession:
        def __init__(self):
            self.closed = False

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            self.closed = True

    session = TrackingSession()
    monkeypatch.setattr(
        "backend.app.db.session.create_session_factory",
        lambda: lambda: session,
    )

    with session_scope() as yielded:
        assert yielded is session

    assert session.closed


def test_database_connectivity_returns_false_for_connection_error(monkeypatch):
    class FailingEngine:
        def connect(self):
            raise OperationalError("SELECT 1", {}, Exception("unreachable"))

    service = DatabaseConnectivityService(FailingEngine())

    assert service.is_reachable() is False


def test_database_connectivity_returns_true_for_select_one(monkeypatch):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, statement):
            return None

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    service = DatabaseConnectivityService(FakeEngine())

    assert service.is_reachable() is True


def test_create_engine_can_be_replaced_without_importing_database(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "analytics")
    monkeypatch.setenv("POSTGRES_USER", "analyst")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")

    engine = create_engine()

    assert engine.url.drivername == "postgresql+psycopg"
