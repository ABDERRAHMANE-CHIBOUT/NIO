import requests
from app.core.config import LLM_MODEL

class LLM:
    def generate(self, prompt: str):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]
