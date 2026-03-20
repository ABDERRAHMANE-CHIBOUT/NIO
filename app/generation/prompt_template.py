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

def build_json_prompt(json_data, question):
    return f"""
            You are an intelligent assistant designed to answer user questions using a provided JSON knowledge base.

            ## CONTEXT
            You are given structured JSON data that contains relevant information.
            You must ONLY rely on this JSON to answer.
            If the answer is not explicitly or implicitly present, say you don't know.

            ## OBJECTIVE
            Answer the user's question accurately, clearly, and concisely.

            ---

            ## REASONING STRATEGY

            1. Parse the JSON carefully:
            - Identify relevant fields, keys, nested structures
            - Handle lists, objects, and partial matches

            2. Match the user question with:
            - Exact information
            - Related or inferred information

            3. Handle special cases:
            - Missing data → say "Information not available"
            - Conflicting entries → mention ambiguity
            - Multiple answers → summarize all possibilities
            - Vague question → interpret reasonably but state assumptions

            ---

            ## RESPONSE FORMAT (STRICT)

            ### 🧾 Answer:
            - Provide a clear and direct answer to the question

            ### 📚 Supporting Evidence:
            - Quote or reference the exact JSON fields or values used
            - Keep it concise but traceable

            ### ⚖️ Confidence Level:
            - HIGH → exact match in JSON
            - MEDIUM → partial or inferred answer
            - LOW → weak connection or missing data

            ---

            ## SELF-VERIFICATION (MANDATORY)

            Before finalizing:
            - Did I strictly use JSON data only?
            - Did I avoid hallucination?
            - Is the answer directly supported by the evidence?
            - If uncertain, did I lower confidence?

            ---

            ## RULES
            - Do NOT invent facts
            - Do NOT use external knowledge
            - Be structured and deterministic
            - Prefer precision over verbosity
            - You can respond to hey, hello..with simple greetings

            ---

            ## INPUT
            User Question:
            {question}

            JSON Data:
            {json_data}
        """