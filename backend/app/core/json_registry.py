from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

REGISTRY_FORMAT_VERSION = 1


class JsonRegistryFile:
    """Atomically persisted list of JSON entries.

    Writes go to a temporary file in the same directory followed by os.replace, so a crash never
    leaves a half-written manifest. Safe for one API process; concurrent writers in separate
    processes can overwrite each other's entries.
    """

    def __init__(self, path_provider: Callable[[], Path]) -> None:
        # Resolved on every call so the location follows the configured upload directory.
        self._path_provider = path_provider

    @property
    def path(self) -> Path:
        return self._path_provider()

    def load(self) -> list[dict[str, Any]]:
        path = self.path
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != REGISTRY_FORMAT_VERSION:
                raise ValueError("Unsupported registry format.")
            entries = payload.get("entries")
            if not isinstance(entries, list):
                raise ValueError("Registry entries must be a list.")
        except (OSError, ValueError) as exc:
            self._quarantine(path, exc)
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    def save(self, entries: list[dict[str, Any]]) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": REGISTRY_FORMAT_VERSION, "entries": entries},
            indent=2,
            sort_keys=True,
        )
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(payload, encoding="utf-8")
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _quarantine(path: Path, exc: Exception) -> None:
        # Keep the unreadable file for inspection instead of silently overwriting it later.
        quarantined = path.with_name(f"{path.name}.corrupt-{time.strftime('%Y%m%d%H%M%S')}")
        try:
            os.replace(path, quarantined)
            logger.error("Registry %s is unreadable (%s); moved it to %s.", path.name, exc, quarantined.name)
        except OSError:
            logger.error("Registry %s is unreadable (%s) and could not be moved aside.", path.name, exc)
