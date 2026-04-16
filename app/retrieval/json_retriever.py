import numpy as np

class JSONRetriever:
    def __init__(self, embedder, json_data):
        self.embedder = embedder

        # flatten JSON into chunks
        self.chunks = []
        for i, item in enumerate(json_data):
            text = str(item)
            emb = self.embedder.embed_query(text)

            self.chunks.append({
                "text": text,
                "embedding": np.asarray(emb, dtype="float32"),
                "metadata": {"source": "json", "id": i}
            })

    def retrieve(self, query, top_k=5):
        query_emb = self.embedder.embed_query(query)
        query_emb = np.asarray(query_emb, dtype="float32")

        scores = []

        for chunk in self.chunks:
            score = np.dot(query_emb, chunk["embedding"])
            scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)

        return [c for _, c in scores[:top_k]]