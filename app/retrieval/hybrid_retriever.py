from collections import defaultdict

class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha=0.5):
        """
        alpha = weight for dense retrieval
        (1 - alpha) = weight for sparse retrieval
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha

    def retrieve(self, query, top_k=5):
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k)

        scores = defaultdict(float)

        # Assign descending rank-based scores
        for rank, doc in enumerate(dense_results):
            scores[doc] += self.alpha * (top_k - rank)

        for rank, doc in enumerate(sparse_results):
            scores[doc] += (1 - self.alpha) * (top_k - rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:top_k]]
