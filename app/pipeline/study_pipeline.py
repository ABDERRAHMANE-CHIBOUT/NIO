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
    - Generates an AI-powered study report
    """

    def __init__(
        self,
        raw_path: str = "data/raw",
        processed_path: str = "data/processed"
    ):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.llm = LLM()

        os.makedirs(self.processed_path, exist_ok=True)

    # -----------------------------
    # STEP 1: LOAD + SPLIT FILES
    # -----------------------------
    def ingest(self):
        """
        Loads raw documents and splits them into chunks.
        """
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
        """
        Converts chunks into structured JSON and saves it.
        Keeps metadata for better traceability.
        """
        data = {
            "documents": [
                {
                    "text": chunk.page_content,
                    "metadata": getattr(chunk, "metadata", {})
                }
                for chunk in chunks
            ]
        }

        json_path = os.path.join(self.processed_path, "study_data.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return data, json_path

    # -----------------------------
    # STEP 3: GENERATE STUDY REPORT
    # -----------------------------
    def generate_study(
        self,
        data: Dict[str, Any],
        domain: str = "general studies",
        provider: str = "QWEN3-30B-A3B-THINKING",
        temperature: float = 0.3,
        max_tokens: int = 1200
    ):
        """
        Uses LLM to transform structured data into a study guide.
        """

        json_text = json.dumps(data, indent=2, ensure_ascii=False)

        prompt = f"""
        You are an expert educator and knowledge synthesizer.

        Your task is to transform structured data into a HIGH-QUALITY STUDY REPORT.

        Domain: {domain}

        Your output MUST include:

        1. Clear Summary
        2. Key Concepts
        3. Rules / Constraints / Principles
        4. Common Mistakes or Risks
        5. A Realistic Case Study
        6. 5 Practice Questions (increasing difficulty)

        Rules:
        - Be structured and clear
        - Do not hallucinate facts outside the data
        - Use only the provided data as source of truth
        - Make it useful for exam preparation

        DATA:
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
        """
        Executes full pipeline: ingest → json → study generation
        """

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
            "mode": "study_generation",
            "num_chunks": len(chunks)
        }