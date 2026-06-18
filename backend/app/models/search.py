"""Search request and response models (module 4.1).

Defines the Pydantic schemas for the ``POST /api/v1/search`` endpoint.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Incoming search request body.

    Example::

        {
            "query": "什么是知识图谱？",
            "top_k": 5
        }
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Natural-language search query",
        examples=["什么是知识图谱？"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of chunks to retrieve",
    )


# ---------------------------------------------------------------------------
# Response items
# ---------------------------------------------------------------------------


class SearchChunk(BaseModel):
    """A single search result chunk."""

    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier (e.g. abc123_chunk_0)",
    )
    content: str = Field(
        ...,
        description="Chunk text content",
    )
    score: float = Field(
        ...,
        description="Cosine similarity score (1.0 = perfect match)",
    )
    doc_id: str = Field(
        ...,
        description="Owning document UUID",
    )
    doc_name: str = Field(
        ...,
        description="Original filename of the owning document",
    )
    doc_type: str = Field(
        ...,
        description="Document type: pdf | md | txt",
    )
    page: int = Field(
        default=1,
        description="Page number within the source document (1-based)",
    )
    chunk_index: int = Field(
        default=0,
        description="Zero-based chunk index within the document",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from Chroma",
    )


class SearchResponse(BaseModel):
    """Search result wrapper returned by ``POST /api/v1/search``."""

    query: str = Field(
        ...,
        description="Original query (echoed back)",
    )
    total_results: int = Field(
        ...,
        description="Number of chunks returned",
    )
    results: list[SearchChunk] = Field(
        default_factory=list,
        description="Ordered list of search result chunks (best first)",
    )
