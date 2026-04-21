
class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # cache dimension once
        self._dim = len(self.model.encode("test"))

    def embed(self, text):
        return self.model.encode(text)

    def get_dimension(self):
        return self._dim