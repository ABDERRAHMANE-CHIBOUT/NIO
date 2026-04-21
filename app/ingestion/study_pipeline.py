import json
from typing import Dict
from app.utils.chunking import smart_chunk


class StudyPipeline:
    def __init__(self, llm, processed_store):
        self.llm = llm
        self.store = processed_store

    def extract_structured_law(self, text: str, doc_name: str) -> Dict:
        prompt = f"""
                You are a legal document parser.

                Extract the following structure STRICTLY in JSON:

                {{
                "doc_name": "{doc_name}",
                "articles": [
                    {{
                    "title": "Article name",
                    "content": "full content",
                    "references": []
                    }}
                ]
                }}

                Rules:
                - Preserve article titles EXACTLY
                - Do NOT merge articles
                - Keep legal structure intact

                TEXT:
                {text}
                """

        result = self.llm.generate(prompt)

        try:
            return json.loads(result)
        except:
            # fallback repair pass
            return {
                "doc_name": doc_name,
                "articles": [
                    {
                        "title": "UNPARSED",
                        "content": result,
                        "references": []
                    }
                ]
            }

    def process_and_store(self, raw_text: str, doc_name: str):
        structured = self.extract_structured_law(raw_text, doc_name)
        self.store.save(doc_name, structured)
        return structured