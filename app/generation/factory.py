from app.generation.llm import LLM


def get_llm(name: str):
    if name == "deepseek":
        return LLM(provider="deepseek")
    elif name == "gpt":
        return LLM(provider="openai")
    else:
        return LLM(provider="default")