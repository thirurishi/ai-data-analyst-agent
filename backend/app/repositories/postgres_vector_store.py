from __future__ import annotations

from sqlalchemy import Column, MetaData, PrimaryKeyConstraint, String, Table, delete, select, text
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import Integer, JSON

from backend.app.repositories.vector_store import (
    VectorStoreDimensionError,
    VectorStoreError,
    VectorStoreRepository,
    _require_single_file_and_model,
)
from backend.app.schemas.embedding import VectorDocumentRecord

# pgvector stores the declared dimension of a vector column in atttypmod.
_EMBEDDING_DIMENSIONS_QUERY = text(
    "SELECT a.atttypmod FROM pg_attribute a JOIN pg_class c ON a.attrelid = c.oid "
    "WHERE c.relname = :table_name AND pg_table_is_visible(c.oid) "
    "AND a.attname = 'embedding' AND NOT a.attisdropped"
)


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
            self._verify_dimensions(connection)
        self._table = table

    def _verify_dimensions(self, connection: Connection) -> None:
        # CREATE TABLE IF NOT EXISTS keeps an existing table, which may use other dimensions.
        existing = connection.execute(_EMBEDDING_DIMENSIONS_QUERY, {"table_name": self.table_name}).scalar()
        if isinstance(existing, int) and existing > 0 and existing != self.dimensions:
            raise VectorStoreDimensionError(
                f"Table '{self.table_name}' stores {existing}-dimension embeddings, "
                f"but EMBEDDING_DIMENSIONS is {self.dimensions}."
            )

    @staticmethod
    def _rows(records: list[VectorDocumentRecord]) -> list[dict]:
        return [
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

    def chunk_signatures(self, *, file_id: str, embedding_model: str) -> dict[str, tuple[str, int]]:
        statement = select(
            self._table.c.chunk_id,
            self._table.c.chunk_text,
            self._table.c.embedding,
        ).where(
            self._table.c.file_id == file_id,
            self._table.c.embedding_model == embedding_model,
        )
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(statement).all()
        except SQLAlchemyError as exc:
            raise VectorStoreError("The PostgreSQL vector store read failed.") from exc
        return {row.chunk_id: (row.chunk_text, len(row.embedding)) for row in rows}

    def replace_file_records(
        self,
        *,
        file_id: str,
        embedding_model: str,
        records: list[VectorDocumentRecord],
    ) -> int:
        _require_single_file_and_model(file_id, embedding_model, records)
        rows = self._rows(records)
        try:
            # One transaction: readers see either the old chunks or the new ones, never a mix.
            with self.engine.begin() as connection:
                connection.execute(
                    delete(self._table).where(
                        self._table.c.file_id == file_id,
                        self._table.c.embedding_model == embedding_model,
                    )
                )
                if rows:
                    statement = postgres_insert(self._table).values(rows)
                    connection.execute(
                        statement.on_conflict_do_update(
                            index_elements=["chunk_id", "embedding_model"],
                            set_={
                                name: statement.excluded[name]
                                for name in ("file_id", "chunk_index", "chunk_text", "embedding", "metadata")
                            },
                        )
                    )
        except SQLAlchemyError as exc:
            raise VectorStoreError("The PostgreSQL vector store write failed.") from exc
        return len(rows)

    def upsert_records(self, records: list[VectorDocumentRecord]) -> int:
        rows = self._rows(records)
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
