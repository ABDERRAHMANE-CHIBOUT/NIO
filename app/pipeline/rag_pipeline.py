from app.ingestion.embedder import Embedder
from app.retrieval.retriever import Retriever
from app.generation.llm import LLM
from app.generation.prompt_template import build_prompt
from app.core.config import EMBEDDING_MODEL


class RAGPipeline:
    def __init__(self, vector_store):

        self.embedder = Embedder(EMBEDDING_MODEL)
        self.vector_store = vector_store
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.llm = LLM()

    def run(self, question: str, doc_ids: list = None, llm_name: str = "Qwen3-30B-A3B-Thinking"):

        chunks = self.retriever.retrieve(
            query=question,
            doc_ids=doc_ids,
            top_k=7
        )

        prompt = build_prompt(chunks, question)

        response = self.llm.generate(
            prompt=prompt,
            provider=llm_name
        )

        return {
            "answer": response,
            "sources": [c["metadata"] for c in chunks]
        }