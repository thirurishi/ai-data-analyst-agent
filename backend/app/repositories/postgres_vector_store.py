from __future__ import annotations

from sqlalchemy import Column, MetaData, PrimaryKeyConstraint, String, Table, insert, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
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
        try:
            with self.engine.begin() as connection:
                result = connection.execute(statement)
                return len(result.scalars().all())
        except SQLAlchemyError as exc:
            raise VectorStoreError("The PostgreSQL vector store write failed.") from exc

    def search(
        self,
        *,
        file_id: str,
        query_vector: list[float],
        top_k: int,
        embedding_model: str,
    ) -> list[VectorDocumentRecord]:
        try:
            from pgvector.sqlalchemy import Vector
        except ImportError as exc:
            raise VectorStoreError("pgvector is required for PostgreSQL vector search.") from exc
        statement = (
            select(self._table.c.chunk_id, self._table.c.file_id, self._table.c.chunk_index,
                   self._table.c.chunk_text, self._table.c.embedding,
                   self._table.c.embedding_model, self._table.c.metadata)
            .where(
                self._table.c.file_id == file_id,
                self._table.c.embedding_model == embedding_model,
            )
            .order_by(self._table.c.embedding.cosine_distance(query_vector))
            .limit(top_k)
        )
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(statement).mappings().all()
        except SQLAlchemyError as exc:
            raise VectorStoreError("The PostgreSQL vector store search failed.") from exc
        return [
            VectorDocumentRecord(
                chunk_id=row["chunk_id"],
                file_id=row["file_id"],
                chunk_index=row["chunk_index"],
                chunk_text=row["chunk_text"],
                embedding=row["embedding"].tolist(),
                embedding_model=row["embedding_model"],
                metadata=row["metadata"],
            )
            for row in rows
        ]
