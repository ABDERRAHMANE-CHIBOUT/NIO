import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings))
        self.text_chunks.extend(texts)

    def search(self, query_embedding, top_k=3):
        distances, indices = self.index.search(
            np.array([query_embedding]), top_k
        )
        return [self.text_chunks[i] for i in indices[0]]
