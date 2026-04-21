from app.generation.llm import LLM

def get_llm(model_name: str):
    return LLM(model_name)