import asyncio
import json
from io import BytesIO

import pytest
from docx import Document
from fastapi import UploadFile
from reportlab.pdfgen import canvas

from backend.app.api.routes.documents import extract_document
from backend.app.api.routes.upload import upload_csv_file
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.services.document_extraction import DocumentExtractionError, DocumentExtractionService


@pytest.fixture
def generated_documents():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    yield paths
    for path in paths:
        path.unlink(missing_ok=True)
    UploadService._file_records.clear()


def _register(
    generated_documents,
    filename: str,
    contents: bytes,
    content_type: str | None = None,
    allowed_extensions: tuple[str, ...] | None = None,
):
    path = UPLOAD_DIR / filename
    path.write_bytes(contents)
    generated_documents.append(path)
    return UploadService.register_uploaded_file(
        path,
        filename,
        content_type,
        allowed_extensions=allowed_extensions,
    )


def _pdf_bytes(text: str) -> bytes:
    output = BytesIO()
    document = canvas.Canvas(output)
    document.setFont("Helvetica", 12)
    document.drawString(72, 720, text)
    document.save()
    return output.getvalue()


def _docx_bytes(paragraphs: list[str]) -> bytes:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)

    output = BytesIO()
    document.save(output)
    return output.getvalue()


def test_txt_extraction_preserves_unicode_and_paragraphs(generated_documents):
    text = "Café naïve\r\n\r\nFirst paragraph.\r\nSecond line."
    record = _register(
        generated_documents,
        "sample71.txt",
        text.encode("utf-8"),
        "text/plain",
        allowed_extensions=(".txt",),
    )

    response = DocumentExtractionService.extract(record.file_id)

    assert response.file_id == record.file_id
    assert response.filename == record.original_filename
    assert response.file_type == "txt"
    assert response.character_count == len(response.extracted_text)
    assert response.page_count is None
    assert response.extracted_text == "Café naïve\n\nFirst paragraph.\nSecond line."
    assert (UPLOAD_DIR / record.saved_filename).read_bytes() == text.encode("utf-8")


def test_pdf_extraction_returns_text_and_page_count(generated_documents):
    record = _register(
        generated_documents,
        "sample71.pdf",
        _pdf_bytes("PDF text"),
        "application/pdf",
        allowed_extensions=(".pdf",),
    )

    response = DocumentExtractionService.extract(record.file_id)

    assert response.file_type == "pdf"
    assert response.page_count == 1
    assert response.extracted_text == "PDF text"


def test_docx_extraction_returns_text_and_metadata(generated_documents):
    record = _register(
        generated_documents,
        "sample71.docx",
        _docx_bytes(["First heading", "Second paragraph"]),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        allowed_extensions=(".docx",),
    )

    response = DocumentExtractionService.extract(record.file_id)

    assert response.file_type == "docx"
    assert response.page_count is None
    assert response.extracted_text == "First heading\n\nSecond paragraph"


def test_empty_document_is_rejected_without_modifying_file(generated_documents):
    record = _register(
        generated_documents,
        "sample71_empty.txt",
        b"",
        "text/plain",
        allowed_extensions=(".txt",),
    )
    original_bytes = (UPLOAD_DIR / record.saved_filename).read_bytes()

    with pytest.raises(DocumentExtractionError, match="empty"):
        DocumentExtractionService.extract(record.file_id)

    assert (UPLOAD_DIR / record.saved_filename).read_bytes() == original_bytes


def test_unsupported_file_type_is_rejected(generated_documents):
    record = _register(
        generated_documents,
        "sample71.csv",
        b"value\n1\n",
        "text/csv",
        allowed_extensions=(".csv",),
    )

    with pytest.raises(DocumentExtractionError, match="Unsupported"):
        DocumentExtractionService.extract(record.file_id)


def test_unknown_file_id_is_rejected():
    with pytest.raises(DocumentExtractionError, match="not found"):
        DocumentExtractionService.extract("00000000-0000-0000-0000-000000000001")


def test_extraction_failure_is_controlled(generated_documents):
    record = _register(
        generated_documents,
        "sample71_corrupt.pdf",
        b"not a valid pdf",
        "application/pdf",
        allowed_extensions=(".pdf",),
    )

    with pytest.raises(DocumentExtractionError, match="could not be extracted"):
        DocumentExtractionService.extract(record.file_id)


def test_route_uses_structured_response_and_safe_file_id(generated_documents):
    record = _register(
        generated_documents,
        "sample71_route.txt",
        b"Hello from the API.\n",
        "text/plain",
        allowed_extensions=(".txt",),
    )

    response = asyncio.run(extract_document(record.file_id))

    assert response.file_id == record.file_id
    assert response.extracted_text == "Hello from the API."


def test_route_rejects_unknown_file_id_without_path_access():
    response = asyncio.run(
        extract_document("00000000-0000-0000-0000-000000000001")
    )

    assert response.status_code == 404
    assert json.loads(response.body)["error"]["code"] == "file_not_found"


@pytest.mark.parametrize(
    ("filename", "content_type", "contents"),
    [
        ("sample71_upload_pdf.pdf", "application/pdf", b"%PDF-1.4\n"),
        ("sample71_upload_docx.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"DOCX"),
        ("sample71_upload_txt.txt", "text/plain", b"Uploaded text"),
    ],
)
def test_upload_route_accepts_supported_document_formats(
    generated_documents, filename, content_type, contents
):
    upload_file = UploadFile(
        file=BytesIO(contents),
        size=len(contents),
        filename=filename,
        headers={"content-type": content_type},
    )

    response = asyncio.run(upload_csv_file(upload_file))

    assert response["file_id"]
    assert response["saved_filename"] == filename
    assert response["content_type"] == content_type
    assert response["file_size"] == len(contents)
    assert (UPLOAD_DIR / filename).read_bytes() == contents
    generated_documents.append(UPLOAD_DIR / filename)
