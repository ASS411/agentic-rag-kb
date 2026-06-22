#!/usr/bin/env python3
"""Redis cache speed benchmark.

Measures the latency improvement of the three-tier cache
(Embedding / Query Rewrite / Retrieval) by comparing cold-path
(cache miss + API) vs warm-path (cache hit) latencies.

Usage::

    # 1. Make sure Redis is running and REDIS_URL is configured:
    #    set REDIS_URL=redis://localhost:6379/0
    #
    # 2. Run benchmark:
    cd backend
    python tests/bench_cache.py

    # 3. To compare: first set REDIS_ENABLED=false and run again
    #    to see the pure no-cache baseline.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Simulated API latency in seconds (adjust to match your real API)
_SIM_API_LATENCY = 0.200   # 200ms per embedding API call
_SIM_LLM_LATENCY = 1.500   # 1.5s per LLM rewrite call

# Test data sizes
_EMB_BATCH_SIZES = [1, 5, 10, 20]
_NUM_ROUNDS = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ensure_redis():
    """Connect to real Redis or return None if unavailable."""
    try:
        from app.core.cache import RedisCacheManager
        cache = RedisCacheManager()
        if await cache._ensure_client():
            return cache
        return None
    except Exception:
        return None


async def _bench_embedder_single(rounds: int = 3):
    """Benchmark single-text embedding cold vs warm."""
    from app.core.cache import cache_key_embedding
    cache = await _ensure_redis()

    text = (
        "RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的深度学习架构。"
    )

    cold_times: list[float] = []
    warm_times: list[float] = []

    for i in range(rounds):
        # ── Cold path: delete cache key, simulate API ──
        if cache:
            await cache.delete(cache_key_embedding(text))
        t0 = time.perf_counter()
        if not cache:
            await asyncio.sleep(_SIM_API_LATENCY)
        else:
            cached = await cache.get(cache_key_embedding(text))
            if cached is None:
                await asyncio.sleep(_SIM_API_LATENCY)
                await cache.set(cache_key_embedding(text), [0.1] * 1024, ttl=3600)
            vector = cached
        cold_times.append(time.perf_counter() - t0)

        # ── Warm path: key already in Redis ──
        t0 = time.perf_counter()
        if not cache:
            await asyncio.sleep(_SIM_API_LATENCY)
        else:
            vector = await cache.get(cache_key_embedding(text))
        warm_times.append(time.perf_counter() - t0)

    return {
        "cold_avg_ms": round(statistics.mean(cold_times) * 1000, 2),
        "warm_avg_ms": round(statistics.mean(warm_times) * 1000, 2),
        "speedup": (
            f"{statistics.mean(cold_times) / statistics.mean(warm_times):.1f}x"
            if statistics.mean(warm_times) > 0 else "N/A"
        ),
    }


async def _bench_embedder_batch(batch_size: int, rounds: int = 2):
    """Benchmark batch embedding cold vs warm."""
    from app.core.cache import cache_key_embedding
    cache = await _ensure_redis()

    texts = [
        f"这是一段用于测试嵌入向量的中文文本，序号为 {i}。"
        f"包含足够多的字符来模拟真实的文档片段。"
        for i in range(batch_size)
    ]
    keys = [cache_key_embedding(t) for t in texts]

    cold_times: list[float] = []
    warm_times: list[float] = []

    for _ in range(rounds):
        # ── Cold: delete all keys ──
        if cache:
            for k in keys:
                await cache.delete(k)

        t0 = time.perf_counter()
        if not cache:
            await asyncio.sleep(_SIM_API_LATENCY * min(batch_size, 5))
        else:
            hits = 0
            for i, k in enumerate(keys):
                val = await cache.get(k)
                if val is not None:
                    hits += 1
            if hits < len(keys):
                await asyncio.sleep(_SIM_API_LATENCY * min(len(keys) - hits, 5))
                for i, k in enumerate(keys):
                    await cache.set(k, [0.1] * 1024, ttl=3600)
        cold_times.append(time.perf_counter() - t0)

        # ── Warm: all keys present ──
        t0 = time.perf_counter()
        if not cache:
            await asyncio.sleep(_SIM_API_LATENCY * min(batch_size, 5))
        else:
            for k in keys:
                await cache.get(k)
        warm_times.append(time.perf_counter() - t0)

    cold_avg = statistics.mean(cold_times) * 1000
    warm_avg = statistics.mean(warm_times) * 1000
    return cold_avg, warm_avg


async def _bench_rewrite(rounds: int = 3):
    """Benchmark query rewrite cold vs warm."""
    from app.core.cache import cache_key_rewrite
    cache = await _ensure_redis()

    question = "什么是检索增强生成（RAG），它相比传统LLM有哪些优势？"

    cold_times: list[float] = []
    warm_times: list[float] = []

    for _ in range(rounds):
        # ── Cold ──
        if cache:
            await cache.delete(cache_key_rewrite(question))
        t0 = time.perf_counter()
        if not cache:
            await asyncio.sleep(_SIM_LLM_LATENCY)
        else:
            cached = await cache.get(cache_key_rewrite(question))
            if cached is None:
                await asyncio.sleep(_SIM_LLM_LATENCY)
                queries = ["RAG定义", "RAG优势", "RAG与传统LLM对比"]
                await cache.set(cache_key_rewrite(question), queries, ttl=3600)
        cold_times.append(time.perf_counter() - t0)

        # ── Warm ──
        t0 = time.perf_counter()
        if not cache:
            await asyncio.sleep(_SIM_LLM_LATENCY)
        else:
            _ = await cache.get(cache_key_rewrite(question))
        warm_times.append(time.perf_counter() - t0)

    return {
        "cold_avg_ms": round(statistics.mean(cold_times) * 1000, 2),
        "warm_avg_ms": round(statistics.mean(warm_times) * 1000, 2),
        "speedup": (
            f"{statistics.mean(cold_times) / statistics.mean(warm_times):.1f}x"
            if statistics.mean(warm_times) > 0 else "N/A"
        ),
    }


async def _bench_retrieve(rounds: int = 3):
    """Benchmark retrieval result caching cold vs warm."""
    from app.core.cache import cache_key_retrieve, _hash_params
    cache = await _ensure_redis()

    queries = ["什么是RAG", "RAG架构", "检索增强生成原理"]
    params_hash = _hash_params(top_k_recall=20, top_k_rerank=5, rerank=True, hybrid=True)
    query_key = "|".join(sorted(queries))
    rk = cache_key_retrieve(query_key, params_hash)

    # Simulated retrieval result (would normally have embedding + chroma + bm25 + reranker latency)
    _SIM_RETRIEVE_LATENCY = 0.500 + _SIM_API_LATENCY  # embedding + overhead

    cold_times: list[float] = []
    warm_times: list[float] = []

    for _ in range(rounds):
        # ── Cold ──
        if cache:
            await cache.delete(rk)
        t0 = time.perf_counter()
        if not cache:
            await asyncio.sleep(_SIM_RETRIEVE_LATENCY)
        else:
            cached = await cache.get(rk)
            if cached is None:
                await asyncio.sleep(_SIM_RETRIEVE_LATENCY)
                await cache.set(rk, {"chunks": [], "total_recalled": 3}, ttl=3600)
        cold_times.append(time.perf_counter() - t0)

        # ── Warm ──
        t0 = time.perf_counter()
        if not cache:
            await asyncio.sleep(_SIM_RETRIEVE_LATENCY)
        else:
            _ = await cache.get(rk)
        warm_times.append(time.perf_counter() - t0)

    return {
        "cold_avg_ms": round(statistics.mean(cold_times) * 1000, 2),
        "warm_avg_ms": round(statistics.mean(warm_times) * 1000, 2),
        "speedup": (
            f"{statistics.mean(cold_times) / statistics.mean(warm_times):.1f}x"
            if statistics.mean(warm_times) > 0 else "N/A"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("  Redis Cache Performance Benchmark")
    print("=" * 72)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Check Redis connectivity
    cache = loop.run_until_complete(_ensure_redis())
    has_redis = cache is not None

    print(f"\n  Redis status: {'CONNECTED' if has_redis else 'NOT CONNECTED (using simulated latency)'}")
    print(f"  Simulated API latency: {_SIM_API_LATENCY*1000:.0f}ms (embedding)")
    print(f"  Simulated LLM latency: {_SIM_LLM_LATENCY*1000:.0f}ms (rewrite)\n")

    # ── 1. Single Embedding ──
    print("  [1] Single Text Embedding Cache")
    print(f"  {'─'*50}")
    result = loop.run_until_complete(_bench_embedder_single(_NUM_ROUNDS))
    print(f"  Cold path avg:  {result['cold_avg_ms']:>8} ms  "
          f"(API call every time)")
    print(f"  Warm path avg:  {result['warm_avg_ms']:>8} ms  "
          f"(Redis hit)")
    print(f"  Speedup:        {result['speedup']:>8}")
    print()

    # ── 2. Batch Embedding ──
    print("  [2] Batch Embedding Cache (per-text granularity)")
    print(f"  {'─'*50}")
    print(f"  {'Batch':>8}  {'Cold(ms)':>10}  {'Warm(ms)':>10}  {'Speedup':>10}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
    for bs in _EMB_BATCH_SIZES:
        cold_ms, warm_ms = loop.run_until_complete(
            _bench_embedder_batch(bs)
        )
        speedup = f"{cold_ms / warm_ms:.1f}x" if warm_ms > 0 else "N/A"
        print(f"  {bs:>8}  {cold_ms:>10.1f}  {warm_ms:>10.1f}  {speedup:>10}")
    print()

    # ── 3. Query Rewrite ──
    print("  [3] Query Rewrite Cache (LLM call)")
    print(f"  {'─'*50}")
    result = loop.run_until_complete(_bench_rewrite(_NUM_ROUNDS))
    print(f"  Cold path avg:  {result['cold_avg_ms']:>8} ms  "
          f"(LLM call every time)")
    print(f"  Warm path avg:  {result['warm_avg_ms']:>8} ms  "
          f"(Redis hit)")
    print(f"  Speedup:        {result['speedup']:>8}")
    print()

    # ── 4. Retrieval Result ──
    print("  [4] Retrieval Result Cache (embedding + Chroma + BM25 + reranker)")
    print(f"  {'─'*50}")
    result = loop.run_until_complete(_bench_retrieve(_NUM_ROUNDS))
    print(f"  Cold path avg:  {result['cold_avg_ms']:>8} ms  "
          f"(full retrieval pipeline)")
    print(f"  Warm path avg:  {result['warm_avg_ms']:>8} ms  "
          f"(Redis hit, JSON deserialize)")
    print(f"  Speedup:        {result['speedup']:>8}")
    print()

    # ── 5. End-to-End summary ──
    print("  " + "=" * 50)
    print("  Cost Savings Estimate (per question, repeat ask)")
    print("  " + "─" * 50)
    print(f"  Without cache:  ~{_SIM_API_LATENCY * 1000:.0f}ms (embedding) "
          f"+ {_SIM_LLM_LATENCY * 1000:.0f}ms (rewrite) "
          f"+ {(_SIM_API_LATENCY + 0.5) * 1000:.0f}ms (retrieve)")
    total_cold = _SIM_API_LATENCY + _SIM_LLM_LATENCY + (_SIM_API_LATENCY + 0.5)
    print(f"                 = {total_cold * 1000:.0f} ms total")

    if has_redis:
        # Measure real Redis read latency
        t0 = time.perf_counter()
        loop.run_until_complete(cache.get("bench:ping"))
        redis_rtt = time.perf_counter() - t0
        total_warm = redis_rtt * 3 + 0.002  # 3 Redis reads + json deserialize
        print(f"  With cache:     ~{redis_rtt * 1000:.1f}ms x 3 (Redis reads)")
        print(f"                 = ~{total_warm * 1000:.1f} ms total")
        print(f"  Savings:        ~{(total_cold - total_warm) * 1000:.0f} ms "
              f"({total_cold / total_warm:.0f}x faster)")
    else:
        print(f"  (Connect Redis to measure real Redis RTT)")

    print()
    print("  " + "=" * 50)
    print("  Tips")
    print("  " + "─" * 50)
    print("  1. Run with REDIS_ENABLED=false to see no-cache baseline")
    print("  2. Adjust _SIM_API_LATENCY / _SIM_LLM_LATENCY to match")
    print("     your actual API response times")
    print("  3. For real end-to-end measurement, use backend logs:")
    print("     grep 'Embedding API call OK' logs | look at elapsed=X.XXs")
    print("     grep 'Rewrite cache hit' logs to see cache hits")
    print("     grep 'Retrieve cache hit' logs to see retrieve cache hits")
    print()

    if cache:
        loop.run_until_complete(cache.close())


if __name__ == "__main__":
    main()
