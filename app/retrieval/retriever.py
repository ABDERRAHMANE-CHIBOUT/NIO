import faiss
import numpy as np
from typing import List
from app.utils.chunking import smart_chunk


class Retriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.metadata = []

    def build_index(self, processed_docs: List[dict]):
        texts = []
        self.metadata = []

        for doc in processed_docs:
            for article in doc["articles"]:
                chunks = smart_chunk(article["content"], article["title"])

                for c in chunks:
                    texts.append(c)
                    self.metadata.append({
                        "doc": doc["doc_name"],
                        "article": article["title"]
                    })

        embeddings = self.embedder.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(embeddings)

        self.texts = texts

    def search(self, query: str, k: int = 5):
        q_emb = np.array([self.embedder.encode(query)]).astype("float32")

        distances, indices = self.index.search(q_emb, k)

        results = []
        for i in indices[0]:
            results.append({
                "text": self.texts[i],
                "meta": self.metadata[i]
            })

        return results