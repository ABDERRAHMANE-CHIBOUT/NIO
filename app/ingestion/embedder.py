from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts)
