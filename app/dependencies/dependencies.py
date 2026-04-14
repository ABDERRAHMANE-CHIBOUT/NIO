from pathlib import Path
import numpy as np

from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import FAISSVectorStore
from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents

# 🔥 SINGLETON INSTANCES
_embedder = None
_vector_store = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_vector_store():
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    BASE_DIR = Path(__file__).resolve().parents[2]
    data_path = BASE_DIR / "data" / "raw"

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    embedder = get_embedder()

    # -----------------------------
    # 📄 Load documents (NEW WAY)
    # -----------------------------
    documents = load_documents(str(data_path))

    if not documents:
        raise ValueError("No documents loaded")

    # -----------------------------
    # ✂️ Split
    # -----------------------------
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No chunks generated")

    # -----------------------------
    # Prepare data
    # -----------------------------
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # -----------------------------
    # ⚡ Embeddings
    # -----------------------------
    embeddings = np.array(embedder.embed(texts)).astype("float32")

    if len(embeddings.shape) != 2:
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")

    dim = embeddings.shape[1]

    # -----------------------------
    # 🧠 FAISS init
    # -----------------------------
    _vector_store = FAISSVectorStore(dim)

    _vector_store.add(
        embeddings=embeddings,
        texts=texts,
        metadatas=metadatas
    )

    print(f"✅ Vector store ready | chunks={len(texts)} | dim={dim}")

    return _vector_store