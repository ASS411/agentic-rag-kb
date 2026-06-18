"""Tests for the Markdown parser (task 2.3).

Covers:
- Heading structure preservation ([H1], [H2], …)
- Inline formatting stripping (bold, italic, links, images)
- Code blocks and inline code
- Tables → readable text
- Empty files, file-not-found, and edge cases
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.core.parsers.md_parser import MdParserError, parse_markdown
from app.models.document import DocType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_md(content: str) -> Path:
    """Write *content* to a temp .md file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".md", delete=False)
    tmp.write(content.encode("utf-8"))
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Heading preservation
# ---------------------------------------------------------------------------


class TestHeadings:
    def test_h1_to_h6_markers(self):
        """All six heading levels get [H1] … [H6] markers."""
        content = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6
"""
        path = _write_md(content)
        try:
            doc = parse_markdown(path)
            text = doc.full_text
            assert "[H1] Level 1" in text
            assert "[H2] Level 2" in text
            assert "[H3] Level 3" in text
            assert "[H4] Level 4" in text
            assert "[H5] Level 5" in text
            assert "[H6] Level 6" in text
        finally:
            path.unlink(missing_ok=True)

    def test_headings_with_body_text(self):
        """Headings followed by paragraphs are preserved in order."""
        content = """# 概述

这是一段正文。

## 方法

具体实现如下。
"""
        path = _write_md(content)
        try:
            doc = parse_markdown(path)
            text = doc.full_text
            # Headings are marked
            assert "[H1] 概述" in text
            assert "[H2] 方法" in text
            # Body text follows
            assert "这是一段正文" in text
            assert "具体实现如下" in text
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Inline formatting stripping
# ---------------------------------------------------------------------------


class TestInlineFormatting:
    def test_bold_and_italic(self):
        """**bold** and *italic* are stripped to plain text."""
        content = "这是 **粗体** 文字，还有 *斜体* 和 ***粗斜体***。"
        path = _write_md(content)
        try:
            doc = parse_markdown(path)
            text = doc.full_text
            assert "**" not in text
            assert "*" not in text
            assert "粗体" in text
            assert "斜体" in text
            assert "粗斜体" in text
        finally:
            path.unlink(missing_ok=True)

    def test_links_and_images(self):
        """Links → link text preserved, URL stripped. Images produce no text."""
        content = (
            "Check the [documentation](https://example.com/doc) for details.\n\n"
            "![logo](https://example.com/logo.png)"
        )
        path = _write_md(content)
        try:
            doc = parse_markdown(path)
            text = doc.full_text
            assert "https://example.com/doc" not in text  # URL stripped
            assert "documentation" in text  # link text preserved
            # Images (<img>) are void elements — BeautifulSoup.get_text()
            # does not return alt text.  This is expected behaviour.
            assert "https://example.com/logo.png" not in text  # img URL stripped
        finally:
            path.unlink(missing_ok=True)

    def test_inline_code(self):
        """`inline code` preserves content, drops backticks."""
        content = "使用 `print('hello')` 函数。"
        path = _write_md(content)
        try:
            doc = parse_markdown(path)
            text = doc.full_text
            assert "`" not in text
            assert "print('hello')" in text
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Code blocks
# ---------------------------------------------------------------------------


class TestCodeBlocks:
    def test_fenced_code_block(self):
        """Fenced code blocks preserve code text, drop fences."""
        content = """```python
def hello():
    print("Hello, World!")
```
"""
        path = _write_md(content)
        try:
            doc = parse_markdown(path)
            text = doc.full_text
            assert "```" not in text
            assert 'def hello():' in text
            assert 'print("Hello, World!")' in text
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TestTables:
    def test_simple_table(self):
        """Markdown tables become readable plain text."""
        content = """| 姓名 | 年龄 | 城市 |
|------|------|------|
| 张三 | 25   | 北京 |
| 李四 | 30   | 上海 |
"""
        path = _write_md(content)
        try:
            doc = parse_markdown(path)
            text = doc.full_text
            assert "姓名" in text
            assert "张三" in text
            assert "北京" in text
            assert "李四" in text
            assert "上海" in text
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_file(self):
        """Empty .md file returns Document with empty text."""
        path = _write_md("")
        try:
            doc = parse_markdown(path)
            assert doc.doc_type == DocType.MARKDOWN
            assert doc.page_count == 1
            assert doc.pages[0].text == ""
            assert doc.char_count == 0
        finally:
            path.unlink(missing_ok=True)

    def test_whitespace_only(self):
        """Whitespace-only file returns empty text."""
        path = _write_md("\n\n   \n\t\n")
        try:
            doc = parse_markdown(path)
            assert doc.char_count == 0
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found(self):
        """Nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_markdown("/tmp/nonexistent_md_12345.md")

    def test_only_headings(self):
        """File with headings only — markers are present."""
        content = """# Title
## Subtitle
"""
        path = _write_md(content)
        try:
            doc = parse_markdown(path)
            assert "[H1] Title" in doc.full_text
            assert "[H2] Subtitle" in doc.full_text
        finally:
            path.unlink(missing_ok=True)

    def test_document_metadata(self):
        """Document metadata includes source_path."""
        path = _write_md("# Test")
        try:
            doc = parse_markdown(path)
            assert doc.metadata["source_path"] == str(path)
        finally:
            path.unlink(missing_ok=True)

    def test_markdown_extension(self):
        """.markdown extension also works."""
        content = "# Hello Markdown"
        tmp = tempfile.NamedTemporaryFile(
            suffix=".markdown", delete=False
        )
        tmp.write(content.encode("utf-8"))
        tmp.close()
        path = Path(tmp.name)
        try:
            doc = parse_markdown(path)
            assert "[H1] Hello Markdown" in doc.full_text
        finally:
            path.unlink(missing_ok=True)
