"""Tests for core/citation.py - regex markers + fallback string matching."""

from __future__ import annotations

from app.core.citation import CitationParser
from app.models.search import SearchChunk


def _make_chunk(idx: int, content: str, doc_name: str = "test.pdf", page: int = 1, score: float = 0.9) -> SearchChunk:
    return SearchChunk(
        chunk_id=f"doc_chunk_{idx}",
        content=content,
        score=score,
        doc_id="doc",
        doc_name=doc_name,
        doc_type="pdf",
        page=page,
        chunk_index=idx,
        metadata={"rerank_score": score},
    )


class TestParseChunkMarkers:
    def test_single_chunk1_marker(self):
        pool = [_make_chunk(0, "RAG stands for Retrieval-Augmented Generation.")]
        answer = "RAG is important [chunk_1]."
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 1
        assert result.orphan_count == 0
        assert result.citations[0].doc_name == "test.pdf"
        assert result.citations[0].reference_label == "[chunk_1]"

    def test_multiple_markers(self):
        pool = [
            _make_chunk(0, "RAG combines retrieval with generation."),
            _make_chunk(1, "Agentic RAG adds reasoning loops."),
            _make_chunk(2, "Vector databases store embeddings."),
        ]
        answer = "RAG [chunk_1] can be agentic [chunk_2], using vector DBs [chunk_3]."
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 3
        assert result.citations[0].reference_label == "[chunk_1]"
        assert result.citations[1].reference_label == "[chunk_2]"
        assert result.citations[2].reference_label == "[chunk_3]"

    def test_chinese_source_marker(self):
        pool = [_make_chunk(0, "Retrieval-Augmented Generation.")]
        answer = "According to the docs [来源 1], RAG is useful."
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 1

    def test_chinese_source_marker_no_space(self):
        pool = [_make_chunk(0, "RAG technique.")]
        answer = "See [来源1] for details."
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 1

    def test_marker_with_underscore(self):
        pool = [_make_chunk(0, "content")]
        answer = "Ref [chunk_1] here."
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 1

    def test_marker_with_dash(self):
        pool = [_make_chunk(0, "content")]
        answer = "Ref [chunk-1] here."
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 1

    def test_out_of_range_marker_is_orphan(self):
        pool = [_make_chunk(0, "only one")]
        answer = "Ref [chunk_5] and [chunk_10]."
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 0
        assert result.orphan_count == 2

    def test_scores_sorted_descending(self):
        pool = [
            _make_chunk(0, "Low relevance", score=0.3),
            _make_chunk(1, "High relevance", score=0.95),
            _make_chunk(2, "Medium relevance", score=0.6),
        ]
        answer = "[chunk_1] [chunk_2] [chunk_3]"
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 3
        scores = [c.score for c in result.citations]
        assert scores == [0.95, 0.6, 0.3]

    def test_no_markers_triggers_fallback(self):
        pool = [_make_chunk(0, "nothing referenced")]
        answer = "Just text, no citations."
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 0
        assert result.fallback_used is True

    def test_duplicate_markers_deduplicated(self):
        pool = [_make_chunk(0, "RAG basics")]
        answer = "[chunk_1] then again [chunk_1]"
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 1

    def test_empty_answer(self):
        pool = [_make_chunk(0, "something")]
        answer = ""
        result = CitationParser().parse(answer, pool)
        assert result.cited_chunk_ids == set()
        assert result.referenced_count == 0

    def test_empty_context_pool(self):
        pool: list[SearchChunk] = []
        answer = "[chunk_1]"
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 0


class TestFallbackStringMatch:
    def test_fallback_detects_verbatim_snippet(self):
        pool = [_make_chunk(0, "The quick brown fox jumps over the lazy dog. This is additional context.")]
        answer = "Here is my answer: The quick brown fox jumps over the lazy dog."
        result = CitationParser().parse(answer, pool)
        assert result.fallback_used is True
        assert result.referenced_count >= 1

    def test_fallback_not_triggered_on_no_match(self):
        pool = [_make_chunk(0, "Completely unrelated content about bananas and apples.")]
        answer = "Answer about quantum physics and black holes."
        result = CitationParser().parse(answer, pool)
        assert result.referenced_count == 0

    def test_regex_takes_priority_over_fallback(self):
        pool = [_make_chunk(0, "RAG was introduced by Lewis et al. in 2020.")]
        answer = "RAG was [chunk_1] introduced."
        result = CitationParser().parse(answer, pool)
        assert result.fallback_used is False
        assert result.referenced_count == 1

    def test_short_snippets_ignored(self):
        pool = [_make_chunk(0, "short")]
        answer = "This answer contains the word short but it's too brief."
        result = CitationParser().parse(answer, pool)
        assert result.cited_chunk_ids == set()
