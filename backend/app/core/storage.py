"""File storage service — raw document persistence on the local filesystem.

Provides:
- ``FileStorage`` — a simple service class for saving and deleting uploaded
  document files under ``data/uploads/{doc_id}/``.

The storage layout is:

    {upload_dir}/
      {doc_id}/
        {original_filename}
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from app.config import settings


class FileStorage:
    """Manages the raw file store for uploaded documents.

    Each document gets its own subdirectory keyed by ``doc_id``.
    The original filename is preserved inside that directory.

    Usage::

        storage = FileStorage()
        path = storage.save(doc_id="abc123", filename="report.pdf", content=b"...")
        storage.delete("abc123")
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or settings.upload.upload_dir)

    # ── Public API ──────────────────────────────────────────────────────────

    def save(self, doc_id: str, filename: str, content: bytes) -> Path:
        """Persist a raw file and return the absolute path where it was written.

        Args:
            doc_id: Unique document identifier (UUID hex).
            filename: Original filename (basename only, no directory traversal).
            content: Raw file bytes.

        Returns:
            The absolute ``Path`` to the saved file.

        Raises:
            OSError: If directory creation or file write fails.
        """
        dir_path = self._doc_dir(doc_id)
        dir_path.mkdir(parents=True, exist_ok=True)

        safe_name = Path(filename).name  # strip any path components
        file_path = dir_path / safe_name
        file_path.write_bytes(content)

        logger.debug("Saved file: {}", file_path)
        return file_path

    def delete(self, doc_id: str) -> bool:
        """Remove the entire storage directory for *doc_id*.

        Returns ``True`` if the directory existed and was removed,
        ``False`` if it did not exist.
        """
        import shutil

        dir_path = self._doc_dir(doc_id)
        if not dir_path.exists():
            return False
        shutil.rmtree(dir_path)
        logger.info("Deleted storage directory: {}", dir_path)
        return True

    def exists(self, doc_id: str) -> bool:
        """Check whether a storage directory exists for *doc_id*."""
        return self._doc_dir(doc_id).exists()

    def get_path(self, doc_id: str) -> Path:
        """Return the storage directory path for *doc_id* (does not create it)."""
        return self._doc_dir(doc_id)

    # ── Internal ────────────────────────────────────────────────────────────

    def _doc_dir(self, doc_id: str) -> Path:
        return self.base_dir / doc_id


# ── Module-level singleton ───────────────────────────────────────────────────

_storage: FileStorage | None = None


def get_storage() -> FileStorage:
    """Return the module-level ``FileStorage`` singleton."""
    global _storage
    if _storage is None:
        _storage = FileStorage()
    return _storage
