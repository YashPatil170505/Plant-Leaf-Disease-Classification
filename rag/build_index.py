"""
build_index.py - Embed knowledge_base/*.md and save the retrieval index
Usage:
    python -m rag.build_index
Re-run whenever a knowledge base file changes.
"""

from collections import Counter
from rag.retriever import build_index, INDEX_DIR

if __name__ == "__main__":
    chunks = build_index()
    per_class = Counter(c["class_name"] for c in chunks)
    n_sources = len({s["url"] for c in chunks for s in c["sources"]})
    print(f"\nIndexed {len(chunks)} chunks from {len(per_class)} documents "
          f"({n_sources} unique sources) -> {INDEX_DIR}/")
    no_cite = [c["id"] for c in chunks if not c["sources"]]
    if no_cite:
        print(f"WARNING: {len(no_cite)} chunks cite no source:")
        for cid in no_cite:
            print("  ", cid)
