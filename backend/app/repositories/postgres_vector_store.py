from __future__ import annotations

from sqlalchemy import Column, MetaData, PrimaryKeyConstraint, String, Table, insert
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.engine import Engine
from sqlalchemy.types import Integer, JSON

from backend.app.repositories.vector_store import VectorStoreError, VectorStoreRepository
from backend.app.schemas.embedding import VectorDocumentRecord


class PostgresVectorStore(VectorStoreRepository):
    def __init__(self, engine: Engine, dimensions: int, table_name: str = "document_chunks") -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")
        try:
            from pgvector.sqlalchemy import Vector
        except ImportError as exc:
            raise VectorStoreError("pgvector is required for PostgreSQL vector storage.") from exc
        self.engine = engine
        self.dimensions = dimensions
        self.table_name = table_name
        self._vector_type = Vector(dimensions)
        self._create_table()

    def _create_table(self) -> None:
        metadata = MetaData()
        table = Table(
            self.table_name,
            metadata,
            Column("chunk_id", String, nullable=False),
            Column("file_id", String, nullable=False),
            Column("chunk_index", Integer, nullable=False),
            Column("chunk_text", String, nullable=False),
            Column("embedding", self._vector_type, nullable=False),
            Column("embedding_model", String, nullable=False),
            Column("metadata", JSON, nullable=False),
            PrimaryKeyConstraint("chunk_id", "embedding_model"),
        )
        with self.engine.begin() as connection:
            table.create(connection, checkfirst=True)
        self._table = table

    def upsert_records(self, records: list[VectorDocumentRecord]) -> int:
        rows = [
            {
                "chunk_id": record.chunk_id,
                "file_id": record.file_id,
                "chunk_index": record.chunk_index,
                "chunk_text": record.chunk_text,
                "embedding": record.embedding,
                "embedding_model": record.embedding_model,
                "metadata": record.metadata,
            }
            for record in records
        ]
        if not rows:
            return 0
        statement = (
            postgres_insert(self._table)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["chunk_id", "embedding_model"])
            .returning(self._table.c.chunk_id)
        )
        with self.engine.begin() as connection:
            result = connection.execute(statement)
            return len(result.scalars().all())
