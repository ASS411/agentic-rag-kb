#!/usr/bin/env python3
"""Reranker performance benchmark (task 1.3).

Measures CPU/GPU inference latency for the BGE-Reranker-v2-m3 cross-encoder
across varying numbers of candidate chunks.  Produces a baseline that can be
compared after future GPU upgrades.

Usage::

    cd backend
    python tests/bench_reranker.py

The script loads the real model (no mocking) and prints a summary table.
"""

from __future__ import annotations

import statistics
import time

from app.core.reranker import Reranker

# ---------------------------------------------------------------------------
# Sample chunks (simulate typical RAG recall results)
# ---------------------------------------------------------------------------

_SAMPLE_QUESTION = "什么是检索增强生成（RAG）？"

_SAMPLE_CHUNKS: list[str] = [
    "RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的深度学习架构。"
    "其核心思想是在大语言模型生成回答之前，先从一个外部知识库中检索相关文档片段，"
    "将检索到的上下文与用户问题一起输入模型，从而提高生成内容的准确性、时效性和可解释性。",

    "RAG 的提出解决了 LLM 的几个关键问题：幻觉、知识截止日期和领域知识不足。"
    "通过检索外部知识，RAG 可以在不重新训练模型的情况下引入最新信息。",

    "RAG 系统通常包含两个核心组件：检索器（Retriever）和生成器（Generator）。"
    "检索器负责从向量数据库中召回候选文档，生成器则基于召回的上下文生成最终答案。",

    "向量数据库如 Chroma、Milvus、Pinecone 等是 RAG 系统的重要基础设施。"
    "它们通过将文本转换为高维向量并计算余弦相似度来实现语义搜索。",

    "Embedding 模型的选择对 RAG 系统的检索质量至关重要。"
    "常用的嵌入模型包括 OpenAI text-embedding-3、BGE 系列和 E5 系列。",

    "Reranker（重排序器）是 RAG 系统中的可选但重要的组件。"
    "与双塔模型不同，Cross-encoder 可以同时编码问题和文档，"
    "捕捉更细粒度的交互信息，从而显著提升检索精度。",

    "RAG 的应用场景包括：企业知识库问答、学术文献检索、"
    "客服机器人、法律文书分析、医疗文献搜索等。",

    "RAG 与传统搜索的区别在于：传统搜索返回文档列表，"
    "RAG 则将检索结果与生成能力结合，直接给出综合答案。",

    "在评估 RAG 系统时，通常使用忠实度（Faithfulness）、"
    "答案相关性（Answer Relevance）和上下文相关性（Context Relevance）等指标。",

    "RAG 面临的挑战包括：检索质量不稳定、上下文窗口限制、"
    "多跳推理困难以及检索与生成之间的语义鸿沟。",

    "HyDE（Hypothetical Document Embeddings）是一种改进 RAG 检索的技术，"
    "通过让 LLM 先生成一个假设性答案，再用该答案去检索，"
    "可以缩小问题与文档之间的语义差距。",

    "Self-RAG 是一种让 LLM 在生成过程中自我反思并决定是否检索的框架，"
    "它通过特殊的反思 token 来控制检索时机和生成质量。",

    "知识图谱增强的 RAG（Graph RAG）使用知识图谱替代或补充向量检索，"
    "特别适合处理需要结构化知识和多跳推理的复杂问题。",

    "RAG 系统中的分块策略对最终效果有显著影响。"
    "常见的分块方法包括固定大小分块、递归分块、语义分块和句子窗口分块。",

    "多路检索（Multi-Query Retrieval）技术通过生成多个不同角度的查询，"
    "合并去重后获得更全面的候选文档集，可以显著提高召回率。",
]

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _time_call(reranker: Reranker, n_pairs: int) -> dict:
    """Time a single compute_similarity call with *n_pairs* (question, chunk) pairs."""
    chunks = _SAMPLE_CHUNKS[:n_pairs]
    pairs = [(_SAMPLE_QUESTION, c) for c in chunks]

    t0 = time.perf_counter()
    scores = reranker.compute_similarity(pairs)
    elapsed = time.perf_counter() - t0

    return {
        "n_pairs": len(pairs),
        "total_s": round(elapsed, 3),
        "per_pair_ms": round(elapsed / len(pairs) * 1000, 2),
        "best_score": round(max(scores), 4),
        "worst_score": round(min(scores), 4),
    }


def main() -> None:
    print("=" * 68)
    print("  Reranker Performance Benchmark")
    print(f"  Model: BAAI/bge-reranker-v2-m3 (CPU)")
    print("=" * 68)

    # Load model (first call triggers download & load)
    reranker = Reranker()

    # Warm-up: one small call to trigger lazy loading + CUDA JIT
    print("\n  Loading model and warming up...")
    _time_call(reranker, 3)
    print("  Warm-up complete.\n")

    # Test increasing pair counts
    pair_counts = [1, 3, 5, 10, 15, 20, 30, 50]

    print(f"  {'Pairs':>6}  {'Total (s)':>10}  {'ms/pair':>10}  "
          f"{'Best':>8}  {'Worst':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for n in pair_counts:
        if n > len(_SAMPLE_CHUNKS):
            break
        result = _time_call(reranker, n)
        print(
            f"  {result['n_pairs']:>6}  "
            f"{result['total_s']:>10.3f}  "
            f"{result['per_pair_ms']:>10.2f}  "
            f"{result['best_score']:>8.4f}  "
            f"{result['worst_score']:>8.4f}"
        )

    # Summary statistics for a typical rerank scenario (top_k_recall=20 → rerank top_k=5)
    print(f"\n  --- Rerank (top_k=5, from 15 candidates) ---")
    from app.core.chunker import Chunk
    from app.models.document import DocType

    chunks = [
        Chunk(
            id=f"bench_chunk_{i}",
            content=_SAMPLE_CHUNKS[i],
            doc_id="bench",
            doc_name="bench.md",
            doc_type=DocType.txt,
            page=1,
            chunk_index=i,
            char_count=len(_SAMPLE_CHUNKS[i]),
            metadata={},
        )
        for i in range(min(15, len(_SAMPLE_CHUNKS)))
    ]

    t0 = time.perf_counter()
    top = reranker.rerank(_SAMPLE_QUESTION, chunks, top_k=5)
    elapsed = time.perf_counter() - t0

    print(f"  Candidates: {len(chunks)}")
    print(f"  Returned:   {len(top)}")
    print(f"  Total time: {elapsed:.3f} s")
    for i, c in enumerate(top):
        score = c.metadata.get("rerank_score", float("nan"))
        print(f"  [{i+1}] score={score:.4f}  {c.content[:60]}...")

    print(f"\n  --- Benchmark complete ---")


if __name__ == "__main__":
    main()
