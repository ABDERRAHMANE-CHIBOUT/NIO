def build_study_prompt(extracted_case, laws):

    return f"""
            You are a senior procurement legal expert.

            CLIENT CASE:
            {extracted_case}

            RELEVANT LAWS:
            {laws}

            Perform a full legal compliance study.

            Determine:

            1. Whether this deal/project is legal
            2. Violations
            3. Missing documents
            4. Financial risks
            5. Contract risks
            6. Regulatory issues
            7. Final recommendation

            Final verdict must be:

            APPROVED
            REJECTED
            NEEDS REVISION

            Explain all violations with legal references.
            """