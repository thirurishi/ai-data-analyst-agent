from __future__ import annotations

import asyncio
import json

import pytest

from backend.app.api.routes.documents import chunk_document
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.schemas.document import DocumentChunkResponse, DocumentChunkingResponse
from backend.app.services.document_chunking import (
    DocumentChunkingError,
    DocumentChunkingService,
)


@pytest.fixture
def registered_document():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    text = "One two three four five six seven eight nine ten."
    path = UPLOAD_DIR / "phase72_chunk_source.txt"
    path.write_text(text, encoding="utf-8")
    record = UploadService.register_uploaded_file(
        path,
        "phase72_chunk_source.txt",
        "text/plain",
        allowed_extensions=(".txt",),
    )
    try:
        yield record, text
    finally:
        path.unlink(missing_ok=True)
        UploadService._file_records.clear()


def test_chunks_are_deterministic_and_metadata_correct(registered_document):
    record, text = registered_document

    first = DocumentChunkingService.chunk(record.file_id, text, chunk_size=10, overlap=2)
    second = DocumentChunkingService.chunk(record.file_id, text, chunk_size=10, overlap=2)

    assert first == second
    assert [chunk.chunk_index for chunk in first.chunks] == list(range(len(first.chunks)))
    assert [chunk.file_id for chunk in first.chunks] == [record.file_id] * len(first.chunks)
    assert [chunk.character_count for chunk in first.chunks] == [len(chunk.text) for chunk in first.chunks]
    assert all(chunk.chunk_id for chunk in first.chunks)
    assert all(chunk.text for chunk in first.chunks)
    assert first.chunks[0].text == text[:10]
    assert first.chunks[1].text == text[8:18]


def test_chunk_boundaries_preserve_order_and_coverage(registered_document):
    record, text = registered_document

    response = DocumentChunkingService.chunk(record.file_id, text, chunk_size=12, overlap=3)

    assert [chunk.chunk_index for chunk in response.chunks] == list(range(len(response.chunks)))
    assert response.chunks[0].text == text[:12]
    assert response.chunks[1].text == text[9:21]
    assert response.chunks[-1].text == text[-4:]
    assert "".join(chunk.text for chunk in response.chunks).endswith(text[-4:])
    assert all(chunk.text for chunk in response.chunks)


def test_overlap_does_not_create_duplicate_positions(registered_document):
    record, text = registered_document

    response = DocumentChunkingService.chunk(record.file_id, text, chunk_size=8, overlap=4)

    assert response.chunks[0].text == text[:8]
    assert response.chunks[1].text == text[4:12]
    assert response.chunks[2].text == text[8:16]
    assert response.chunks[-1].text == text[-5:]
    assert "".join(chunk.text for chunk in response.chunks).endswith(text[-5:])
    assert all(chunk.text for chunk in response.chunks)


def test_exact_boundary_and_short_text_are_handled(registered_document):
    record, text = registered_document

    exact = DocumentChunkingService.chunk(record.file_id, "1234567890", chunk_size=10, overlap=0)
    short = DocumentChunkingService.chunk(record.file_id, "short", chunk_size=10, overlap=0)

    assert [chunk.text for chunk in exact.chunks] == ["1234567890"]
    assert [chunk.text for chunk in short.chunks] == ["short"]


def test_empty_text_produces_no_chunks(registered_document):
    record, _ = registered_document

    response = DocumentChunkingService.chunk(record.file_id, "", chunk_size=400, overlap=100)

    assert response.chunks == []


def test_unicode_text_is_not_split(registered_document):
    record, _ = registered_document
    text = "Café naïve: 🌍🚀"

    response = DocumentChunkingService.chunk(record.file_id, text, chunk_size=5, overlap=0)

    assert "".join(chunk.text for chunk in response.chunks) == text
    assert all(chunk.character_count == len(chunk.text) for chunk in response.chunks)


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (-1, 0), (10, 10), (10, 11)])
def test_invalid_configuration_is_rejected(registered_document, chunk_size, overlap):
    record, _ = registered_document

    with pytest.raises(DocumentChunkingError, match="(?i)chunk size|overlap"):
        DocumentChunkingService.chunk(record.file_id, "text", chunk_size, overlap)


def test_unknown_file_id_is_rejected():
    with pytest.raises(DocumentChunkingError, match="not found"):
        DocumentChunkingService.chunk("00000000-0000-0000-0000-000000000001", "text")


def test_route_uses_trusted_file_id_and_structured_response(registered_document):
    record, text = registered_document

    response = asyncio.run(chunk_document(record.file_id, 8, 2))

    assert isinstance(response, DocumentChunkingResponse)
    assert response.file_id == record.file_id
    assert response.chunk_count == len(response.chunks)
    assert response.chunks


def test_route_rejects_unknown_file_id_without_path_access():
    response = asyncio.run(chunk_document("00000000-0000-0000-0000-000000000001", 8, 2))

    assert response.status_code == 404
    assert json.loads(response.body)["error"]["code"] == "file_not_found"


def test_chunk_response_models_validate_metadata(registered_document):
    record, _ = registered_document
    chunk = DocumentChunkResponse(
        chunk_id="chunk-0",
        file_id=record.file_id,
        chunk_index=0,
        text="hello",
        character_count=5,
    )

    assert chunk.character_count == len(chunk.text)
