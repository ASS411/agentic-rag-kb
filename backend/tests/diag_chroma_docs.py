#!/usr/bin/env python3
"""Diagnostic script: list all doc_ids currently in Chroma vs MySQL.

Usage:
    cd backend
    python -m tests.diag_chroma_docs
"""

import asyncio
from app.db.chroma import ChromaStore
from app.db.mysql import async_session_factory
from app.models.document import DocumentModel
from sqlalchemy import select


async def main():
    chroma = ChromaStore()
    total = chroma.count()
    print(f"Chroma total chunks: {total}")

    # Get ALL chunks from Chroma (not via query, use get_all)
    ids, docs, metas = chroma.get_all()

    chroma_doc_ids = {}
    for i, m in zip(ids, metas):
        did = m.get("doc_id", "?")
        dname = m.get("doc_name", "?")
        if did not in chroma_doc_ids:
            chroma_doc_ids[did] = {"name": dname, "count": 0}
        chroma_doc_ids[did]["count"] += 1

    print(f"\n=== Chroma documents ({len(chroma_doc_ids)}):")
    for did, info in chroma_doc_ids.items():
        print(f"  {did[:16]}...  chunks={info['count']:>3}  name={info['name']}")

    # Get valid doc_ids from MySQL
    async with async_session_factory() as session:
        result = await session.execute(select(DocumentModel.doc_id, DocumentModel.file_name))
        rows = result.fetchall()

    mysql_ids = {row[0]: row[1] for row in rows}
    print(f"\n=== MySQL documents ({len(mysql_ids)}):")
    for mid, mname in mysql_ids.items():
        print(f"  {mid[:16]}...  name={mname}")

    # Find stale
    stale = set(chroma_doc_ids.keys()) - set(mysql_ids.keys())
    if stale:
        print(f"\n!!! STALE docs in Chroma but NOT in MySQL ({len(stale)}):")
        for sid in sorted(stale):
            info = chroma_doc_ids.get(sid, {})
            print(f"  {sid[:16]}...  chunks={info.get('count','?')}  name={info.get('name','?')}")
        print("\nRun this to force-clean:")
        print("  python -m tests.force_clean_chroma")
    else:
        print("\nChroma is clean — no stale docs.")


if __name__ == "__main__":
    asyncio.run(main())
