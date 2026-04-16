import json
from app.ingestion.embedder import Embedder
from app.retrieval.retriever import Retriever
from app.generation.llm import LLM
from app.generation.prompt_template import build_prompt
from app.core.config import EMBEDDING_MODEL
from app.utils.json_loader import load_json
from app.retrieval.json_retriever import JSONRetriever

class RAGPipeline:
    def __init__(
        self,
        vector_store,
        json_path: str = "data/processed/data.json"
        
    ):
        self.embedder = Embedder(EMBEDDING_MODEL)
        self.vector_store = vector_store
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.llm = LLM()

        # 🔥 ALWAYS LOAD BASE KNOWLEDGE
        self.base_data = load_json(json_path)
        self.base_text = json.dumps(self.base_data, indent=2, ensure_ascii=False)
        self.json_retriever = JSONRetriever(self.embedder, self.base_data)
    # -----------------------------
    # MAIN RUN
    # -----------------------------
    def run(
    self,
    question: str,
    doc_ids: list = None,
    llm_name: str = "Qwen3-30B-A3B-Thinking"
    ):

        # 🔹 1. (OPTIONAL) Query expansion
        try:
            expanded_query = self.llm.generate(
                prompt=f"Rewrite for legal retrieval: {question}"
            )
            query_used = expanded_query if expanded_query else question
        except:
            query_used = question

        # 🔹 2. Retrieve from JSON (GROUND TRUTH)
        json_chunks = self.json_retriever.retrieve(query_used, top_k=5)

        # 🔹 3. Retrieve from vector DB (RAW FILES)
        vector_chunks = self.retriever.retrieve(
            query=query_used,
            doc_ids=doc_ids,
            top_k=10
        )

        # 🔹 4. Simple rerank (merge)
        all_chunks = json_chunks + vector_chunks

        def rerank(query, chunks):
            return sorted(
                chunks,
                key=lambda c: query.lower() in c["text"].lower(),
                reverse=True
            )

        final_chunks = rerank(question, all_chunks)[:6]

        # 🔹 5. Build context (CLEAN)
        context = "\n\n".join([
            f"[{c['metadata'].get('source', 'doc')}] {c['text']}"
            for c in final_chunks
        ])

        # 🔹 6. Prompt
        prompt = build_prompt(context, question)

        # 🔹 7. Generate
        response = self.llm.generate(
            prompt=prompt,
            provider=llm_name
        )

        return {
            "answer": response,
            "sources": [c["metadata"] for c in final_chunks],
            "mode": "hybrid_rag_v2"
        }