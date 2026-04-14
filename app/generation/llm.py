from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class LLM:
    def __init__(self):
        """
        Multi-provider LLM router (clean + safe)
        """
        self.providers = {}

    # -----------------------------
    # CLIENT CACHE
    # -----------------------------
    def _get_client(self, provider: str):
        if not provider:
            raise ValueError("Provider is required")

        if provider in self.providers:
            return self.providers[provider]

        api_key = os.getenv(f"API_KEY_{provider}")
        base_url = os.getenv(f"BASE_URL_{provider}")

        if not api_key:
            raise ValueError(f"Missing API_KEY_{provider} in environment variables")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.providers[provider] = client
        return client

    # -----------------------------
    # MODEL RESOLUTION (IMPORTANT FIX)
    # -----------------------------
    def _get_model(self, provider: str):
        model = os.getenv(f"LLM_MODEL_{provider}")
        if not model:
            raise ValueError(f"Missing LLM_MODEL_{provider} in environment variables")
        return model

    # -----------------------------
    # GENERATE
    # -----------------------------
    def generate(
        self,
        prompt: str,
        provider: str = "QWEN",
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 800
    ):
        try:
            client = self._get_client(provider)

            # fallback to env model if not provided
            if model is None:
                model = os.getenv(f"MODEL_{provider}")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful and precise assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"