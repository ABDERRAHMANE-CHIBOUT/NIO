import json
import os
from typing import List, Dict

PROCESSED_DIR = "data/processed"


class ProcessedStore:
    def __init__(self, path: str = PROCESSED_DIR):
        self.path = path

    def save(self, doc_name: str, data: Dict):
        os.makedirs(self.path, exist_ok=True)
        file_path = os.path.join(self.path, f"{doc_name}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, doc_names: List[str]) -> List[Dict]:
        docs = []

        for name in doc_names:
            file_path = os.path.join(self.path, f"{name}.json")
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                docs.append(json.load(f))

        return docs

    def list_docs(self):
        return [
            f.replace(".json", "")
            for f in os.listdir(self.path)
            if f.endswith(".json")
        ]