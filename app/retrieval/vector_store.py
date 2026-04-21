import faiss
import numpy as np
from typing import List, Dict, Any


class FAISSVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, embeddings, texts, metadatas):
        embeddings = np.asarray(embeddings, dtype="float32")

        # ensure 2D
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got {embeddings.shape}")

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def has_doc(self, doc_id: str) -> bool:
        return any(m.get("doc_id") == doc_id for m in self.metadatas)

    def search(self, query_embedding, k=5, filter_docs=None):

        query_embedding = np.asarray(query_embedding, dtype="float32")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim == 3:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                f"Dim mismatch: query={query_embedding.shape}, index={self.index.d}"
            )

        # 🔥 FIX 1: normalize filter format
        if isinstance(filter_docs, list):
            filter_docs = {"doc_id": filter_docs}

        # 🔥 FIX 2: increase search space when filtering
        search_k = 50 if filter_docs else k

        scores, indices = self.index.search(query_embedding, search_k)

        results = []

        for i, score in zip(indices[0], scores[0]):
            if i == -1:
                continue

            metadata = self.metadatas[i]

            # 🔥 FIX 3: robust filtering
            if filter_docs and "doc_id" in filter_docs:
                allowed = filter_docs["doc_id"]

                if isinstance(allowed, dict):
                    allowed = allowed.get("$in", [])

                if isinstance(allowed, str):
                    allowed = [allowed]

                if metadata.get("doc_id") not in allowed:
                    continue

            results.append({
                "text": self.texts[i],
                "score": float(score),
                "metadata": metadata
            })

            # 🔥 FIX 4: stop AFTER filtering
            if len(results) >= k:
                break

        return results