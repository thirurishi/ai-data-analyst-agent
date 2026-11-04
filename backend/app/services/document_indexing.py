from __future__ import annotations

from backend.app.schemas.embedding import DocumentIndexResponse
from backend.app.services.document_chunking import DocumentChunkingService
from backend.app.services.document_embeddings import DocumentEmbeddingService
from backend.app.services.document_extraction import DocumentExtractionService


class DocumentIndexingService:
    """Run the extract -> chunk -> embed -> store pipeline for one uploaded document."""

    def __init__(self, embedding_service: DocumentEmbeddingService | None = None) -> None:
        self._embedding_service = embedding_service

    def index(
        self,
        file_id: str,
        chunk_size: int = DocumentChunkingService.DEFAULT_CHUNK_SIZE,
        overlap: int = DocumentChunkingService.DEFAULT_OVERLAP,
    ) -> DocumentIndexResponse:
        extraction = DocumentExtractionService.extract(file_id)
        chunking = DocumentChunkingService.chunk(
            file_id,
            extraction.extracted_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        # Built after file validation so missing or unsupported files are reported first.
        if self._embedding_service is None:
            self._embedding_service = DocumentEmbeddingService()
        embedding = self._embedding_service.embed_chunks(file_id, chunking.chunks)
        return DocumentIndexResponse(
            file_id=file_id,
            filename=extraction.filename,
            file_type=extraction.file_type,
            character_count=extraction.character_count,
            page_count=extraction.page_count,
            chunk_count=embedding.chunk_count,
            stored_count=embedding.stored_count,
            embedding_model=embedding.embedding_model,
            dimensions=embedding.dimensions,
        )
