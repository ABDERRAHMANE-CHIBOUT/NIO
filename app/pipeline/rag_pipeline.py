from app.core.laws_processor import LawsProcessor
from app.generation.prompt_template import build_prompt


class RAGPipeline:

    def __init__(self, llm, vector_store, embedder, laws_processor: LawsProcessor):
        self.llm = llm
        self.vector_store = vector_store
        self.embedder = embedder
        self.laws_processor = laws_processor  # 👈 now REQUIRED

    # -------------------------
    # MAIN PIPELINE
    # -------------------------
    def run(self, question, doc_ids=None, llm_name=None, mode="laws"):

        print(f"🔥 MODE: {mode} | DOC_IDS: {doc_ids}")
        print(f"🧠 LLM PROVIDER: {getattr(self.llm, 'provider', 'unknown')}")

        try:
            query_emb = self.embedder.embed(question)

            # =========================================================
            # MODE 1: LAWS ONLY (NOW FULLY VECTORIZED)
            # =========================================================
            if mode == "laws" or not doc_ids:

                core_knowledge = self.laws_processor.search(query_emb, k=50)

                full_context = self._format_context(
                    core=core_knowledge,
                    retrieved=None
                )

                prompt = build_prompt(full_context, question)

                return {
                    "answer": self.llm.generate(prompt),
                    "mode": "laws",
                    "sources": ["laws_vector_store"]
                }

            # =========================================================
            # MODE 2: HYBRID RAG (DOCS + LAWS)
            # =========================================================
            retrieved_docs = self.vector_store.search(
                query_emb,
                filter_docs=doc_ids
            )

            # fallback if no docs
            if not retrieved_docs:
                print("⚠️ No retrieved docs, fallback to laws only")

                core_knowledge = self.laws_processor.search(query_emb, k=15)

                full_context = self._format_context(
                    core=core_knowledge,
                    retrieved=None
                )

                prompt = build_prompt(full_context, question)

                return {
                    "answer": self.llm.generate(prompt),
                    "mode": "laws_fallback",
                    "sources": ["laws_vector_store_fallback"]
                }

            # laws still act as grounding context
            core_knowledge = self.laws_processor.search(query_emb, k=50)

            full_context = self._format_context(
                core=core_knowledge,
                retrieved=retrieved_docs
            )

            prompt = build_prompt(full_context, question)

            return {
                "answer": self.llm.generate(prompt),
                "mode": "rag",
                "sources": doc_ids
            }

        except Exception as e:
            return {
                "answer": f"⚠️ RAG Error: {str(e)}",
                "mode": "error",
                "sources": []
            }

    # -------------------------
    # CONTEXT BUILDER
    # -------------------------
    def _format_context(self, core, retrieved):

        context_parts = []

        # CORE LAWS (NOW SEMANTIC)
        context_parts.append("[CORE_KNOWLEDGE]")
        context_parts.append(self._safe_serialize(core))

        # RETRIEVED DOCS
        context_parts.append("\n[RETRIEVED_DOCS]")

        if retrieved:
            context_parts.append(self._safe_serialize(retrieved))
        else:
            context_parts.append("EMPTY")

        return "\n".join(context_parts)

    # -------------------------
    # SAFE SERIALIZATION
    # -------------------------
    def _safe_serialize(self, data):
        import json

        try:
            if isinstance(data, (dict, list)):
                return json.dumps(data, indent=2, ensure_ascii=False)
            return str(data)
        except Exception:
            return str(data)