import json
from app.utils.json_loader import load_json
from app.generation.prompt_template import build_json_prompt
from app.generation.llm import LLM


class JSONPipeline:
    def __init__(self, json_path: str = "data/processed/data.json"):
        """
        Loads JSON once at startup
        """
        self.llm = LLM()
        self.data = load_json(json_path)

        # Pre-stringify once (fast + avoids recomputing)
        self.json_text = json.dumps(self.data, indent=2, ensure_ascii=False)

    # -----------------------------
    # MAIN RUN METHOD
    # -----------------------------
    def run(
        self,
        question: str,
        provider: str = "QWEN3-30B-A3B-THINKING",   # ✅ FIXED (your working provider)
        model: str = None,        # optional (comes from env if None)
        temperature: float = 0.3,
        max_tokens: int = 800
    ):
        """
        Ask questions directly on structured JSON data
        """

        # build prompt safely
        prompt = build_json_prompt(self.json_text, question)

        # call LLM (no silent hiding anymore)
        answer = self.llm.generate(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "answer": answer,
            "mode": "json_direct",
            "provider": provider
        }