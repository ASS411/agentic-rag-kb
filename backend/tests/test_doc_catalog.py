"""Tests for DocCatalog (semantic doc-name catalog) and resolve_doc_filter.

Uses a mocked ``Embedder`` so the tests don't hit the real embedding API
and so we can deterministically control the cosine similarities that
drive doc_filter selection.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from app.core.doc_catalog import DocCatalog
from app.core.embedder import Embedder
from app.core.prompts import resolve_doc_filter


class _FakeEmbedder(Embedder):
    """Embedder that returns deterministic unit vectors."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        # Deliberately skip the parent __init__ so we don't try to read
        # settings / API keys.
        self._vectors = vectors

    async def embed(self, text: str) -> list[float]:  # type: ignore[override]
        if text in self._vectors:
            return list(self._vectors[text])
        # Fallback: any text not explicitly registered gets a vector
        # close to the catalog entry it was meant to match.  Callers
        # can override per-test by adding to self._vectors directly.
        return list(self._vectors.get("__default__", [-1.0, 0.0]))

    async def embed_batch(self, texts: Iterable[str]):  # type: ignore[override]
        return [list(self._vectors.get(t, [-1.0, 0.0])) for t in texts]


def _add(vec_a: list[float], vec_b: list[float]) -> list[float]:
    """Return a (roughly) unit-length combination of two unit vectors."""
    out = [a + b for a, b in zip(vec_a, vec_b)]
    norm = sum(x * x for x in out) ** 0.5 or 1.0
    return [x / norm for x in out]


def _make_catalog(
    docs: list[tuple[str, str]],
    *,
    threshold: float = 0.5,
    min_gap: float = 0.01,
) -> tuple[DocCatalog, _FakeEmbedder]:
    """Build a DocCatalog over a temporary collection with a fake embedder.

    ``docs`` is a list of (doc_id, file_name) pairs.  Each file_name is
    assigned a unique orthogonal unit vector on the 2-D plane (so two
    docs have similarity ~0 with each other).  Callers then set
    the question's vector explicitly to control the test outcome.
    """
    vectors: dict[str, list[float]] = {}
    for i, (_, name) in enumerate(docs):
        # Spread entries on a circle so pairwise cosine similarity is
        # roughly 0 and predictable.
        import math
        theta = i * math.pi / max(len(docs), 2)
        vectors[name] = [math.cos(theta), math.sin(theta)]

    embedder = _FakeEmbedder(vectors)
    cat = DocCatalog(
        chroma=__import__("app.db.chroma", fromlist=["ChromaStore"]).ChromaStore(),
        embedder=embedder,  # type: ignore[arg-type]
        threshold=threshold,
        min_gap=min_gap,
        collection_name="doc_catalog_test",
    )
    # Start from a clean slate
    existing = cat._collection.get(include=[])["ids"]
    if existing:
        cat._collection.delete(ids=existing)
    return cat, embedder


class TestDocCatalog:
    @pytest.mark.asyncio
    async def test_add_and_find_relevant(self):
        cat, embedder = _make_catalog([("d1", "agent项目要求.txt")])
        await cat.add("d1", "agent项目要求.txt")

        # Identical question → cos sim 1.0 → resolves to the file
        embedder._vectors["根据agent项目要求"] = list(
            embedder._vectors["agent项目要求.txt"]
        )
        names = await cat.find_relevant("根据agent项目要求")
        assert names == ["agent项目要求.txt"]

    @pytest.mark.asyncio
    async def test_remove(self):
        cat, _ = _make_catalog([("d1", "agent项目要求.txt")])
        await cat.add("d1", "agent项目要求.txt")
        assert cat._collection.count() == 1
        cat.remove("d1")
        assert cat._collection.count() == 0

    @pytest.mark.asyncio
    async def test_re_add_overwrites(self):
        cat, _ = _make_catalog([("d1", "agent项目要求.txt")])
        await cat.add("d1", "agent项目要求.txt")
        await cat.add("d1", "agent项目要求.txt")
        assert cat._collection.count() == 1

    @pytest.mark.asyncio
    async def test_empty_question_returns_empty(self):
        cat, _ = _make_catalog([("d1", "agent项目要求.txt")])
        await cat.add("d1", "agent项目要求.txt")
        assert await cat.find_relevant("") == []
        assert await cat.find_relevant("   ") == []

    @pytest.mark.asyncio
    async def test_empty_catalog_returns_empty(self):
        cat, _ = _make_catalog([])
        assert await cat.find_relevant("任何问题") == []

    @pytest.mark.asyncio
    async def test_low_similarity_returns_empty(self):
        cat, embedder = _make_catalog([
            ("d1", "agent项目要求.txt"),
            ("d2", "checklist.md"),
        ])
        await cat.add("d1", "agent项目要求.txt")
        await cat.add("d2", "checklist.md")

        # Question with a vector opposite to both (cos sim ~ -1 each)
        embedder._vectors["完全不相关的内容"] = [-1.0, 0.0]
        names = await cat.find_relevant("完全不相关的内容")
        assert names == []


class TestResolveDocFilter:
    """The async wrapper used by chat.py."""

    @pytest.mark.asyncio
    async def test_falls_back_to_regex_when_catalog_disabled(self, monkeypatch):
        # Force feature off; the regex extractor should take over.
        from app.config import settings
        monkeypatch.setattr(settings.agent, "doc_catalog_enabled", False)
        names = await resolve_doc_filter("根据agent项目要求.txt")
        assert names == ["agent项目要求.txt"]

    @pytest.mark.asyncio
    async def test_falls_back_to_regex_when_catalog_empty(self):
        # Catalog has no entries → falls through to regex extractor.
        cat, _ = _make_catalog([])
        names = await resolve_doc_filter("根据agent项目要求.txt", cat)
        assert names == ["agent项目要求.txt"]
