from __future__ import annotations

from backend.app.ingestion.upload_service import UploadService
from backend.app.schemas.document import DocumentChunkResponse, DocumentChunkingResponse


class DocumentChunkingError(ValueError):
    pass


class DocumentChunkingService:
    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_OVERLAP = 100

    @staticmethod
    def chunk(
        file_id: str,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> DocumentChunkingResponse:
        if not isinstance(text, str):
            raise DocumentChunkingError("Document text must be a string.")
        if chunk_size <= 0:
            raise DocumentChunkingError("Chunk size must be greater than zero.")
        if overlap < 0 or overlap >= chunk_size:
            raise DocumentChunkingError("Overlap must be non-negative and less than chunk size.")

        try:
            UploadService.resolve_file_id(file_id, (".pdf", ".docx", ".txt"))
        except (FileNotFoundError, ValueError) as exc:
            if isinstance(exc, FileNotFoundError):
                raise DocumentChunkingError("The requested document was not found.") from exc
            raise DocumentChunkingError("The requested document identifier is invalid.") from exc

        chunks: list[DocumentChunkResponse] = []
        step = chunk_size - overlap
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunk_index = len(chunks)
            chunks.append(
                DocumentChunkResponse(
                    chunk_id=f"{file_id}:{chunk_index:06d}",
                    file_id=file_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    character_count=len(chunk_text),
                )
            )
            if end == len(text):
                break
            start += step

        return DocumentChunkingResponse(
            file_id=file_id,
            chunk_count=len(chunks),
            chunks=chunks,
        )
