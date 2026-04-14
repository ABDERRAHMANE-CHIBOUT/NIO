from sentence_transformers import SentenceTransformer
import numpy as np
import torch


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        self.model.max_seq_length = 256

    def embed(self, texts, batch_size: int = 64):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str):
        return self.embed([query])[0]