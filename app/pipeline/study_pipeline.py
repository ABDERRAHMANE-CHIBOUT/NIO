import os

from app.utils.document_manager import DocumentManager
from app.core.laws_processor import LawsProcessor
from app.utils.case_extractor import CaseExtractor
from app.generation.study_prompt import build_study_prompt
from app.ingestion.loader import load_file


class StudyPipeline:

    def __init__(
        self,
        llm,
        embedder,
        laws_processor: LawsProcessor
    ):
        
        self.llm = llm
        self.embedder = embedder
        self.laws_processor = laws_processor
        self.doc_manager = DocumentManager()
        self.extractor = CaseExtractor(llm)

    def run(self, doc_ids):

        try:
            print(f"📚 Running study mode for docs: {doc_ids}")

            # -------------------------
            # Load ALL selected docs
            # -------------------------
            all_docs_content = []

            for doc_id in doc_ids:
                doc = self.doc_manager.get_document(doc_id)

                if not doc:
                    continue

                content = load_file(doc["path"])

                if isinstance(content, list):
                    for page in content:
                        if page["text"].strip():
                            all_docs_content.append(
                                page["text"]
                            )
                else:
                    if content.strip():
                        all_docs_content.append(content)

            if not all_docs_content:
                return {
                    "status": "error",
                    "message": "No valid documents found"
                }

            # -------------------------
            # Extract full case info
            # -------------------------
            extracted_case = self.extractor.extract(
                all_docs_content
            )

            # -------------------------
            # Retrieve laws only
            # -------------------------
            law_query = str(extracted_case)

            query_emb = self.embedder.embed(law_query)

            relevant_laws = self.laws_processor.search(
                query_emb,
                k=50
            )

            # -------------------------
            # Generate final study
            # -------------------------
            prompt = build_study_prompt(
                extracted_case,
                relevant_laws
            )

            final_study = self.llm.generate(prompt)

            return {
                "status": "success",
                "mode": "study",
                "study": final_study,
                "extracted_case": extracted_case,
                "laws_used": relevant_laws
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }