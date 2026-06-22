#!/usr/bin/env python3
"""Force-clean Chroma: remove all chunks whose doc_id is not in MySQL.

Uses Chroma's get_all() instead of query_batch() for 100% coverage.

Usage:
    cd backend
    python -m tests.force_clean_chroma
"""

import asyncio
from app.db.chroma import ChromaStore
from app.db.mysql import async_session_factory
from app.models.document import DocumentModel
from sqlalchemy import select
from loguru import logger


async def main():
    chroma = ChromaStore()
    before_count = chroma.count()
    print(f"Chroma total before: {before_count}")

    # Get ALL doc_ids from Chroma via get_all (not query, guaranteed full)
    _, _, metas = chroma.get_all()
    chroma_doc_ids = set(m.get("doc_id") for m in metas if m.get("doc_id"))

    # Get valid doc_ids from MySQL
    async with async_session_factory() as session:
        result = await session.execute(select(DocumentModel.doc_id))
        mysql_ids = set(row[0] for row in result.fetchall())

    stale = chroma_doc_ids - mysql_ids
    print(f"Valid doc_ids (MySQL): {len(mysql_ids)}")
    print(f"Total doc_ids (Chroma): {len(chroma_doc_ids)}")
    print(f"Stale doc_ids to remove: {len(stale)}")

    removed_total = 0
    for sid in sorted(stale):
        removed = chroma.delete_by_doc_id(sid)
        removed_total += removed
        print(f"  Removed doc_id={sid[:16]}... ({removed} chunks)")

    after_count = chroma.count()
    print(f"\nChroma total after: {after_count}")
    print(f"Total removed: {removed_total} chunks")


if __name__ == "__main__":
    asyncio.run(main())
