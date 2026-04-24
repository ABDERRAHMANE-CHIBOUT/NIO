import json


class CaseExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, documents):
        full_text = "\n\n".join(documents)

        prompt = f"""
            You are a senior legal document analyst.

            Extract ALL information from the following client documents .

            Return STRICT valid JSON only.

            DOCUMENTS:
            {full_text}
            """

        try:
            response = self.llm.generate(prompt)

            try:
                return json.loads(response)

            except:
                return {
                    "raw_extraction": response
                }

        except Exception as e:
            return {
                "error": f"Extraction failed: {str(e)}"
            }