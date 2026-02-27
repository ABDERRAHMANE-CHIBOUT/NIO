def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)

    return f"""
        You are an assistant answering based only on the context below.

        Context:
        {context}

        Question:
        {question}

        Answer clearly and concisely:
        """
