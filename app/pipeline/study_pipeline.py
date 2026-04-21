import os
import json
from typing import List, Tuple, Dict, Any

from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.generation.llm import LLM


class StudyPipeline:
    """
    StudyPipeline:
    - Ingests raw documents
    - Splits into chunks
    - Builds structured JSON dataset
    - Uses LAWS as ground truth
    - Generates an AI-powered study report
    """

    def __init__(
        self,
        raw_path: str = "data/raw",
        processed_path: str = "data/processed",
        laws_path: str = "data/laws/laws.json"
    ):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.laws_path = laws_path
        self.llm = LLM()

        os.makedirs(self.processed_path, exist_ok=True)

        # 🔥 Load laws as ground truth
        if os.path.exists(self.laws_path):
            with open(self.laws_path, "r", encoding="utf-8") as f:
                self.laws = json.load(f)
        else:
            self.laws = {}

    # -----------------------------
    # STEP 1: LOAD + SPLIT FILES
    # -----------------------------
    def ingest(self):
        documents = load_documents(self.raw_path)

        if not documents:
            raise ValueError(f"No documents found in {self.raw_path}")

        chunks = split_documents(documents)

        if not chunks:
            raise ValueError("Document splitting returned empty chunks")

        return chunks

    # -----------------------------
    # STEP 2: BUILD STRUCTURED JSON
    # -----------------------------
    def build_json(self, chunks) -> Tuple[Dict[str, Any], str]:
        data = {
            "documents": [
                {
                    "text": chunk.page_content,
                    "metadata": getattr(chunk, "metadata", {})
                }
                for chunk in chunks
            ]
        }

        json_path = os.path.join(self.processed_path, "data.json")

        # ✅ Save JSON for traceability
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return data, json_path

    # -----------------------------
    # STEP 3: GENERATE STUDY REPORT
    # -----------------------------
    def generate_study(
        self,
        data: Dict[str, Any],
        provider: str = "QWEN3-30B-A3B-THINKING",
        domain: str = "general studies",
        temperature: float = 0.3,
        max_tokens: int = 1200
    ):
        json_text = json.dumps(data, indent=2, ensure_ascii=False)
        laws_text = json.dumps(self.laws, indent=2, ensure_ascii=False)

        prompt = f"""
                You are an expert educator and knowledge synthesizer.

                Your task is to generate a HIGH-QUALITY STUDY REPORT grounded in OFFICIAL LAWS.

                ---

                ## PRIORITY OF SOURCES

                1. LAWS (STRICT GROUND TRUTH - MUST BE FOLLOWED)
                2. DOCUMENTS (SUPPORTING CONTEXT)

                If documents contradict laws → IGNORE documents.

                ---

                ## OBJECTIVE

                Transform the data into a structured study guide aligned with the laws.
                domain: {domain}

                ---

                ## OUTPUT MUST INCLUDE:

                1. Clear Summary (aligned with laws)
                2. Key Concepts (based on laws)
                3. Rules / Constraints / Principles (STRICTLY from laws)
                4. Common Mistakes or Violations (based on laws)
                5. A Realistic Case Study (consistent with laws)

                ---

                ## RULES

                - Use LAWS as the primary source of truth
                - Use DOCUMENTS only to enrich or illustrate
                - Do NOT invent information
                - If something is missing in laws → say "Not specified in laws"
                - Be precise and structured

                ---

                ## LAWS:
                {laws_text}

                ---

                ## DOCUMENT DATA:
                {json_text}
                """

        response = self.llm.generate(
            prompt=prompt,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response

    # -----------------------------
    # FULL PIPELINE
    # -----------------------------
    def run(
        self,
        domain: str = "general studies",
        provider: str = "QWEN3-30B-A3B-THINKING",
        temperature: float = 0.3,
        max_tokens: int = 1200
    ):
        chunks = self.ingest()

        data, json_path = self.build_json(chunks)

        study = self.generate_study(
            data=data,
            domain=domain,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "study": study,
            "json_path": json_path,
            "domain": domain,
            "mode": "study_generation",
            "num_chunks": len(chunks)
        }