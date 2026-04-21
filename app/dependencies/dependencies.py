# app/dependencies/dependencies.py

from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import FAISSVectorStore
from app.core.processed_store import ProcessedStore
from app.ingestion.study_pipeline import StudyPipeline
from app.core.laws_processor import LawsProcessor




# -------------------------
# Singletons (global state)
# -------------------------

_embedder = None
_vector_store = None
_processed_store = None
_study_pipeline = None
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


# -------------------------
# Processed Store (JSON / structured data)
# -------------------------
def get_processed_store():
    global _processed_store

    if _processed_store is None:
        _processed_store = ProcessedStore()

    return _processed_store


# -------------------------
# Study Pipeline (LLM + processed store)
# -------------------------
def get_study_pipeline(llm):
    """
    Creates or returns a cached StudyPipeline instance.
    NOTE: pipeline is tied to LLM instance.
    """
    global _study_pipeline

    if _study_pipeline is None:
        _study_pipeline = StudyPipeline(
            llm=llm,
            processed_store=get_processed_store()
        )

    return _study_pipeline


def get_laws_processor():
    global _laws_processor

    if _laws_processor is None:
        embedder = get_embedder()

        _laws_processor = LawsProcessor(embedder=embedder)

        # 🔥 IMPORTANT: build once at initialization
        _laws_processor.build()

        print("⚖️ LawsProcessor initialized and indexed")

    return _laws_processor