# app/dependencies/dependencies.py

from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import FAISSVectorStore
from app.core.laws_processor import LawsProcessor
from app.pipeline.ingestion_pipeline import IngestionPipeline
from app.utils.document_manager import DocumentManager

doc_manager = DocumentManager()

def get_ingestion_pipeline():
    return IngestionPipeline(get_embedder(), get_vector_store())

def get_document_manager():
    return doc_manager

# -------------------------
# Singletons (global state)
# -------------------------

_embedder = None
_vector_store = None
_laws_processor = None

# -------------------------
# Embedder
# -------------------------
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


# -------------------------
# Vector Store (FAISS)
# -------------------------
def get_vector_store():
    global _vector_store

    if _vector_store is None:
        embedder = get_embedder()

        dim = len(embedder.embed("test"))

        _vector_store = FAISSVectorStore(dim=dim)

    return _vector_store


def get_laws_processor():
    global _laws_processor

    if _laws_processor is None:
        embedder = get_embedder()

        _laws_processor = LawsProcessor(embedder=embedder)

        # 🔥 IMPORTANT: build once at initialization
        _laws_processor.build()

        print("⚖️ LawsProcessor initialized and indexed")

    return _laws_processor