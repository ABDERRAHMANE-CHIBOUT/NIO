from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
#from app.retrieval.sparse_retriever import SparseRetriever
#from app.retrieval.hybrid_retriever import HybridRetriever
#from app.generation.hallucination_guard import HallucinationGuard
from app.generation.llm import LLM
from app.generation.prompt_template import build_prompt
from app.core.config import EMBEDDING_MODEL

class RAGPipeline:
    def __init__(self):
        # Load and process documents once at startup
        raw_docs = load_documents("data/raw")
        split_docs = split_documents(raw_docs)

        texts = [doc.page_content for doc in split_docs]

        self.embedder = Embedder(EMBEDDING_MODEL)
        embeddings = self.embedder.embed(texts)

        dimension = len(embeddings[0])
        self.vector_store = VectorStore(dimension)
        self.vector_store.add(embeddings, texts)

        self.retriever = Retriever(self.embedder, self.vector_store)

        """
        # Sparse retriever
        self.sparse_retriever = SparseRetriever(texts)

        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.retriever,
            sparse_retriever=self.sparse_retriever,
            alpha=0.6  # we can tune this experimentally
        )

        # Detecte hallucination
        self.hallucination_guard = HallucinationGuard(
            embedder=self.embedder,
            threshold=0.65
        )
        """
        self.llm = LLM()

    def run(self, question: str):

        chunks = self.retriever.retrieve(question)
        #chunks = self.hybrid_retriever.retrieve(question)
        prompt = build_prompt(chunks, question)
        return self.llm.generate(prompt)
        """
        guard_result = self.hallucination_guard.check(answer, chunks)

        if not guard_result["grounded"]:
            return {
                "answer": "⚠️ The system is uncertain about this answer based on retrieved context.",
                "confidence_score": guard_result["similarity"],
                "grounded": False
            }

        return {
            "answer": answer,
            "confidence_score": guard_result["similarity"],
            "grounded": True
        }

        """
