from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class LLM:
    def __init__(self, provider: str):
        """
        Multi-provider LLM router
        provider = QWEN / GPT_OSS_120B / etc.
        """
        self.provider = provider
        self.clients = {}
        

    # -----------------------------
    # CLIENT CACHE
    # -----------------------------
    def _get_client(self):
        if self.provider in self.clients:
            return self.clients[self.provider]

        api_key = os.getenv(f"API_KEY_{self.provider}")
        base_url = os.getenv(f"BASE_URL_{self.provider}")
        
        print(f"🧬 Using provider: {self.provider}")
        print(f"🔑 API KEY FOUND: {bool(api_key)}")
        print(f"🌐 BASE URL: {base_url}")

        if not api_key:
            raise ValueError(f"Missing API_KEY_{self.provider}")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.clients[self.provider] = client
        return client

    # -----------------------------
    # MODEL RESOLUTION
    # -----------------------------
    def _get_model(self):
        model = os.getenv(f"MODEL_{self.provider}")
        
        if not model:
            raise ValueError(f"Missing LLM_MODEL_{self.provider}")
        return model

    # -----------------------------
    # GENERATE
    # -----------------------------
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096
    ):
        try:
            client = self._get_client()
            model = self._get_model()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful and precise legal assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"