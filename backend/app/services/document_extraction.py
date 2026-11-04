from __future__ import annotations

import re
from pathlib import Path

from docx import Document as WordDocument
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from backend.app.ingestion.upload_service import UploadService
from backend.app.schemas.document import DocumentExtractionResponse


class DocumentExtractionError(ValueError):
    pass


class DocumentExtractionService:
    SUPPORTED_TYPES = {
        "pdf": ".pdf",
        "docx": ".docx",
        "txt": ".txt",
    }

    @staticmethod
    def extract(file_id: str) -> DocumentExtractionResponse:
        record = UploadService.get_file_record_by_id(file_id)
        if record is None:
            try:
                UploadService.resolve_file_id(file_id)
            except FileNotFoundError as exc:
                raise DocumentExtractionError("The requested document was not found.") from exc
            except ValueError as exc:
                raise DocumentExtractionError("The requested document identifier is invalid.") from exc
            raise DocumentExtractionError("The requested document identifier is invalid.")

        file_type = DocumentExtractionService._file_type(Path(record.saved_filename))
        if file_type not in DocumentExtractionService.SUPPORTED_TYPES:
            raise DocumentExtractionError("Unsupported document file type.")

        try:
            file_path = UploadService.resolve_file_id(file_id, tuple(DocumentExtractionService.SUPPORTED_TYPES.values()))
        except (FileNotFoundError, ValueError) as exc:
            if isinstance(exc, FileNotFoundError):
                raise DocumentExtractionError("The requested document was not found.") from exc
            raise DocumentExtractionError("The requested document identifier is invalid.") from exc
        if not file_path.stat().st_size:
            raise DocumentExtractionError("The document is empty.")

        try:
            extracted_text, page_count = DocumentExtractionService._extract(file_path, file_type)
        except (OSError, ValueError, RuntimeError, TypeError, PdfReadError) as exc:
            raise DocumentExtractionError("The document could not be extracted.") from exc

        normalized_text = DocumentExtractionService._normalize_text(extracted_text)
        if not normalized_text.strip():
            raise DocumentExtractionError("The document contains no extractable text.")

        record = UploadService.get_file_record_by_id(file_id)
        return DocumentExtractionResponse(
            file_id=file_id,
            filename=record.original_filename if record is not None else file_path.name,
            file_type=file_type,
            character_count=len(normalized_text),
            page_count=page_count,
            extracted_text=normalized_text,
        )

    @staticmethod
    def _file_type(file_path: Path) -> str:
        suffix = file_path.suffix.lower().lstrip(".")
        return {
            ".pdf": "pdf",
            ".docx": "docx",
            ".txt": "txt",
        }.get(f".{suffix}", suffix)

    @staticmethod
    def _extract(file_path: Path, file_type: str) -> tuple[str, int | None]:
        if file_type == "txt":
            raw_text = file_path.read_text(encoding="utf-8-sig", errors="strict")
            return raw_text, None
        if file_type == "pdf":
            reader = PdfReader(str(file_path))
            text_parts = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(text_parts), len(reader.pages)
        if file_type == "docx":
            document = WordDocument(str(file_path))
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            return "\n\n".join(paragraphs), None
        raise DocumentExtractionError("Unsupported document file type.")

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        return text.strip("\n")
