class Retriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedder.embed([query])[0]
        return self.vector_store.search(query_embedding, top_k)
