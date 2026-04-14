import numpy as np


class Retriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, doc_ids=None, top_k=5):

        query_embedding = self.embedder.embed_query(query)
        query_embedding = np.asarray(query_embedding, dtype="float32")

        # fix shape ALWAYS
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim == 3:
            query_embedding = query_embedding.reshape(1, -1)

        metadata_filter = None

        if doc_ids:
            metadata_filter = {
                "doc_id": doc_ids[0] if len(doc_ids) == 1 else {"$in": doc_ids}
            }

        return self.vector_store.search(
            query_embedding=query_embedding,
            k=top_k,
            filter=metadata_filter
        )