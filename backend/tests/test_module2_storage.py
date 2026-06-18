"""Tests for the FileStorage service (task 2.7)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from app.core.storage import FileStorage


class TestFileStorage:
    """FileStorage unit tests."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = FileStorage(base_dir=self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_save_creates_file(self):
        """save() writes content and returns the correct path."""
        path = self.storage.save("abc123", "report.pdf", b"%PDF-test")
        assert path.exists()
        assert path.read_bytes() == b"%PDF-test"
        assert path.parent.name == "abc123"

    def test_save_strips_path_traversal(self):
        """Filename is sanitised — '../' is stripped."""
        path = self.storage.save("abc123", "../../etc/passwd", b"x")
        assert path.name == "passwd"  # basename only
        assert "../../" not in str(path)

    def test_save_preserves_original_filename(self):
        """The original filename is kept inside the doc_id directory."""
        path = self.storage.save("doc1", "论文.pdf", "中文".encode("utf-8"))
        assert path.name == "论文.pdf"
        assert path.parent.name == "doc1"

    def test_delete_removes_directory(self):
        """delete() removes the entire doc directory."""
        self.storage.save("abc", "f.txt", b"x")
        assert self.storage.exists("abc")

        result = self.storage.delete("abc")
        assert result is True
        assert not self.storage.exists("abc")

    def test_delete_nonexistent_returns_false(self):
        """delete() on nonexistent doc_id returns False."""
        result = self.storage.delete("no-such-id")
        assert result is False

    def test_exists(self):
        """exists() correctly reports presence."""
        assert not self.storage.exists("xyz")
        self.storage.save("xyz", "a.txt", b"x")
        assert self.storage.exists("xyz")

    def test_get_path(self):
        """get_path() returns correct path without creating it."""
        p = self.storage.get_path("myid")
        assert p.name == "myid"
        assert not p.exists()  # does not create
        assert p.parent == Path(self.storage.base_dir)
