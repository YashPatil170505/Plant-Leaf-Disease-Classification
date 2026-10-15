"""
retriever.py - Dense retrieval over the knowledge base

Embeds chunks with a local sentence-transformers model and keeps the vectors in
a small on-disk index (rag_index/). With a few hundred chunks an exact cosine
search in numpy is fast enough, so a separate vector database is not needed.

The CNN's prediction is used as a metadata filter: when we know (or suspect)
which classes the leaf belongs to, we search only those classes' chunks.
"""

import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.knowledge import load_chunks, embedding_text, GENERAL_CLASS

EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX_DIR    = "rag_index"


def build_index(kb_dir="knowledge_base", index_dir=INDEX_DIR):
    chunks = load_chunks(kb_dir)
    encoder = SentenceTransformer(EMBED_MODEL)
    vecs = encoder.encode([embedding_text(c) for c in chunks],
                          normalize_embeddings=True, show_progress_bar=True)
    os.makedirs(index_dir, exist_ok=True)
    np.save(os.path.join(index_dir, "embeddings.npy"), vecs.astype(np.float32))
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump({"embed_model": EMBED_MODEL, "chunks": chunks}, f, indent=1)
    return chunks


class Retriever:
    def __init__(self, index_dir=INDEX_DIR):
        with open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.chunks  = data["chunks"]
        self.vecs    = np.load(os.path.join(index_dir, "embeddings.npy"))
        self.encoder = SentenceTransformer(data["embed_model"])
        self.kb_classes = {c["class_name"] for c in self.chunks}

    def resolve_classes(self, class_names):
        """Map predicted classes to KB classes; healthy/unknown classes use the general doc."""
        resolved = [c for c in class_names if c in self.kb_classes]
        if len(resolved) < len(class_names) and GENERAL_CLASS in self.kb_classes:
            resolved.append(GENERAL_CLASS)
        return resolved

    def search(self, query, k=4, class_filter=None):
        """Return the top-k chunks (with scores). class_filter=None searches everything."""
        t0 = time.perf_counter()
        q = self.encoder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        scores = self.vecs @ q

        idx = np.arange(len(self.chunks))
        if class_filter:
            allowed = set(self.resolve_classes(class_filter))
            idx = np.array([i for i in idx if self.chunks[i]["class_name"] in allowed], dtype=int)
            if idx.size == 0:
                idx = np.arange(len(self.chunks))

        top = idx[np.argsort(-scores[idx])[:k]]
        results = [dict(self.chunks[i], score=float(scores[i])) for i in top]
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return results
