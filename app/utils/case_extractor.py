import json


class CaseExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, documents):
        full_text = "\n\n".join(documents)

        prompt = f"""
                You are a senior legal procurement analyst specialized in contract review,
                public procurement documentation, and compliance investigations.

                Your role is to extract ALL factual information from client documents.

                IMPORTANT:
                - Extract facts ONLY
                - Do NOT perform legal analysis
                - Do NOT determine violations
                - Do NOT infer legal conclusions
                - Do NOT invent missing data

                If information is missing:
                use null

                Return STRICT VALID JSON ONLY.

                -----------------------------------------
                DOCUMENTS
                -----------------------------------------
                {full_text}

                -----------------------------------------
                REQUIRED JSON STRUCTURE
                -----------------------------------------

                {{
                    "project_information": {{
                        "project_name": null,
                        "project_type": null,
                        "project_value": null,
                        "currency": null,
                        "procurement_method": null,
                        "project_location": null
                    }},

                    "timeline": {{
                        "publication_date": null,
                        "submission_deadline": null,
                        "award_date": null,
                        "contract_signature_date": null,
                        "delivery_deadline": null
                    }},

                    "parties": {{
                        "client_entity": null,
                        "winning_company": null,
                        "competing_bidders": [],
                        "internal_committees": [],
                        "decision_makers": []
                    }},

                    "submitted_documents": {{
                        "available_documents": [],
                        "missing_documents": [],
                        "invalid_documents": []
                    }},

                    "procurement_phase": {{
                        "publication_issues": [],
                        "evaluation_issues": [],
                        "eligibility_issues": [],
                        "transparency_issues": [],
                        "conflict_of_interest_signals": []
                    }},

                    "contract_execution": {{
                        "delivery_status": null,
                        "technical_issues": [],
                        "payment_issues": [],
                        "contract_modifications": [],
                        "penalty_issues": []
                    }},

                    "financial_information": {{
                        "payments_made": null,
                        "advance_payments": null,
                        "budget_overrun": null
                    }},

                    "risk_flags": [
                        "fraud",
                        "conflict_of_interest",
                        "document_missing",
                        "delivery_failure",
                        "payment_irregularity"
                    ],

                    "raw_key_events": []
                }}

                -----------------------------------------
                STRICT RULES
                -----------------------------------------

                - Return JSON ONLY
                - No markdown
                - No explanations
                - No legal reasoning
                - No article references
                - Preserve factual neutrality
                - Missing fields = null
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