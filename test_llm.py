from app.generation.llm import LLM

def main():
    llm = LLM()

    prompt = """
    Write a short explanation of how a vector database works in machine learning.
    """

    response = llm.generate(
        prompt=prompt,
        provider="Qwen3-30B-A3B-Thinking"
    )

    print("\n===== QWEN RESPONSE =====\n")
    print(response)


if __name__ == "__main__":
    main()