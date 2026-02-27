from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HallucinationGuard:
    def __init__(self, embedder, threshold=0.65):
        """
        threshold: minimum similarity to consider grounded
        """
        self.embedder = embedder
        self.threshold = threshold

    def check(self, answer, context_chunks):
        context_text = " ".join(context_chunks)

        embeddings = self.embedder.embed([answer, context_text])

        answer_vec = embeddings[0].reshape(1, -1)
        context_vec = embeddings[1].reshape(1, -1)

        similarity = cosine_similarity(answer_vec, context_vec)[0][0]

        is_grounded = similarity >= self.threshold

        return {
            "similarity": float(similarity),
            "grounded": is_grounded
        }
