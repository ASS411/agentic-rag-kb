"""Integration tests for Module 2.1 — Document upload endpoint.

Uses FastAPI ``TestClient`` so no live server or MySQL is needed.
"""

from __future__ import annotations

import io
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def override_upload_dir():
    """Replace the upload directory with a temp dir for the duration of the test."""
    with tempfile.TemporaryDirectory() as tmp:
        original = settings.upload.upload_dir
        settings.upload.upload_dir = tmp
        yield tmp
        settings.upload.upload_dir = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file(filename: str, content: bytes | str) -> tuple[str, io.BytesIO, str]:
    """Create an in-memory file for upload testing.

    Returns ``(filename, file_obj, content_type)``.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return filename, io.BytesIO(content), ""


def _content_type_for(filename: str) -> str:
    """Return appropriate content-type per extension."""
    ext = os.path.splitext(filename)[1].lower()
    mapping = {
        ".pdf": "application/pdf",
        ".md": "text/markdown",
        ".txt": "text/plain",
    }
    return mapping.get(ext, "application/octet-stream")


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------


class TestUploadSuccess:
    """Happy-path upload scenarios."""

    def test_upload_pdf(self, override_upload_dir):
        """Upload a minimal PDF file — should succeed with 201."""
        filename = "test.pdf"
        content = b"%PDF-1.4\n%faux PDF content\n%%EOF"
        files = {"file": (filename, io.BytesIO(content), _content_type_for(filename))}

        resp = client.post("/api/v1/documents/upload", files=files)

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["code"] == 200  # APIResponse uses 200 even for 201 http

        data = body["data"]
        assert data["file_name"] == "test.pdf"
        assert data["doc_type"] == "pdf"
        assert data["size_bytes"] == len(content)
        assert len(data["doc_id"]) == 32  # UUID hex
        assert data["page_count"] == 0  # not yet parsed
        assert data["chunk_count"] == 0  # not yet chunked

    def test_upload_markdown_mime(self, override_upload_dir):
        """Upload via 'text/markdown' MIME type."""
        content = "# Hello\n\nWorld"
        filename = "readme.md"
        files = {"file": (filename, io.BytesIO(content.encode()), _content_type_for(filename))}

        resp = client.post("/api/v1/documents/upload", files=files)
        assert resp.status_code == 201
        assert resp.json()["data"]["doc_type"] == "md"

    def test_upload_txt_extension_fallback(self, override_upload_dir):
        """Upload with octet-stream MIME — should fall back to extension."""
        content = b"plain text content"
        filename = "notes.txt"
        files = {"file": (filename, io.BytesIO(content), "application/octet-stream")}

        resp = client.post("/api/v1/documents/upload", files=files)
        assert resp.status_code == 201
        assert resp.json()["data"]["doc_type"] == "txt"


class TestUploadValidation:
    """Edge cases and error scenarios."""

    def test_empty_file(self, override_upload_dir):
        """Empty files should be rejected with 400."""
        filename = "empty.pdf"
        files = {"file": (filename, io.BytesIO(b""), _content_type_for(filename))}

        resp = client.post("/api/v1/documents/upload", files=files)
        assert resp.status_code == 400
        assert "empty" in resp.json()["message"].lower()

    def test_unsupported_type(self, override_upload_dir):
        """.exe files should be rejected with 400."""
        filename = "malware.exe"
        files = {"file": (filename, io.BytesIO(b"x"), "application/octet-stream")}

        resp = client.post("/api/v1/documents/upload", files=files)
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["message"].lower()

    def test_size_exceeds_limit(self, override_upload_dir):
        """Files larger than max_upload_size_mb are rejected with 413."""
        # Temporarily lower the limit for this test
        original_max = settings.upload.max_upload_size_mb
        settings.upload.max_upload_size_mb = 1  # 1 MB
        try:
            # Create ~1.1 MB of data
            big_content = b"x" * (1 * 1024 * 1024 + 100 * 1024)
            filename = "big.txt"
            files = {"file": (filename, io.BytesIO(big_content), _content_type_for(filename))}

            resp = client.post("/api/v1/documents/upload", files=files)
            assert resp.status_code == 413
            assert "exceeds" in resp.json()["message"].lower()
        finally:
            settings.upload.max_upload_size_mb = original_max

    def test_no_filename(self, override_upload_dir):
        """Missing filename — FastAPI rejects at framework level (422)."""
        files = {"file": ("", io.BytesIO(b"x"), "text/plain")}
        resp = client.post("/api/v1/documents/upload", files=files)
        assert resp.status_code == 422

    def test_missing_file_field(self, override_upload_dir):
        """Request without the 'file' field — FastAPI returns 422."""
        resp = client.post("/api/v1/documents/upload")
        assert resp.status_code == 422

    def test_case_insensitive_extension(self, override_upload_dir):
        """Extension matching is case-insensitive."""
        filename = "Report.PDF"
        files = {"file": (filename, io.BytesIO(b"%PDF-x"), _content_type_for(filename))}

        resp = client.post("/api/v1/documents/upload", files=files)
        assert resp.status_code == 201
        assert resp.json()["data"]["doc_type"] == "pdf"


class TestListAndDelete:
    """List / delete stubs behave correctly."""

    def test_list_documents_empty(self):
        """GET /documents returns empty list (stub)."""
        resp = client.get("/api/v1/documents?page=1&size=10")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["items"] == []
        assert body["data"]["total"] == 0

    def test_delete_nonexistent_is_idempotent(self):
        """DELETE on nonexistent doc_id returns 200 OK."""
        resp = client.delete("/api/v1/documents/fake-id-12345")
        assert resp.status_code == 200
        assert resp.json()["data"] == "fake-id-12345"
