class StudyPipeline:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, topic: str, doc_ids: list, llm):
        chunks = self.retriever.retrieve(topic, doc_ids=doc_ids)

        context = "\n".join(chunks)

        prompt = f"""
                You are an expert tutor.

                Context:
                {context}

                Task:
                1. Explain the topic simply
                2. Extract key points
                3. Give examples
                4. Suggest a mini quiz

                Topic: {topic}
                """

        return llm.generate(prompt)