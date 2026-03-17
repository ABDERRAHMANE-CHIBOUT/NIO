from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class LLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

    def generate(self, prompt: str):
        response = self.client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content