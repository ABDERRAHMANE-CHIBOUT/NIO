from app.utils.json_loader import load_json
from app.generation.prompt_template import build_json_prompt
from app.generation.llm import LLM
import json

class JSONPipeline:
    def __init__(self):
        self.llm = LLM()
        self.data = load_json("data/processed/data.json")
        self.json_text = json.dumps(self.data, indent=2)

    def run(self, question: str):
        prompt = build_json_prompt(self.json_text, question)
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "mode": "json_direct"
        }