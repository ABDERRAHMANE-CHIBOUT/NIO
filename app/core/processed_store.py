import os
import json
from typing import List, Dict, Optional


class ProcessedStore:
    """
    Manages structured documents stored in /data/processed.

    Each file = one structured document produced by the LLM:
    {
        "doc_name": "...",
        "articles": [...]
    }
    """

    def __init__(self, base_path: str = "data/processed"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    # ----------------------------
    # 💾 SAVE PROCESSED DOCUMENT
    # ----------------------------
    def save(self, doc_name: str, data: Dict) -> str:
        """
        Save structured LLM output into /processed.
        """
        file_path = os.path.join(self.base_path, f"{doc_name}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return file_path

    # ----------------------------
    # 📥 LOAD ONE DOCUMENT
    # ----------------------------
    def load_one(self, doc_name: str) -> Optional[Dict]:
        """
        Load a single processed document.
        """
        file_path = os.path.join(self.base_path, f"{doc_name}.json")

        if not os.path.exists(file_path):
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----------------------------
    # 📚 LOAD MULTIPLE DOCS
    # ----------------------------
    def load(self, doc_names: List[str]) -> List[Dict]:
        """
        Load multiple processed documents (used by RAG).
        Only selected docs are loaded → important for scoped embedding.
        """
        docs = []

        for name in doc_names:
            doc = self.load_one(name)
            if doc:
                docs.append(doc)

        return docs

    # ----------------------------
    # 📜 LIST AVAILABLE DOCS
    # ----------------------------
    def list_docs(self) -> List[str]:
        """
        Returns all processed document names (without .json).
        """
        return [
            f.replace(".json", "")
            for f in os.listdir(self.base_path)
            if f.endswith(".json")
        ]

    # ----------------------------
    # 🧹 DELETE DOC (optional utility)
    # ----------------------------
    def delete(self, doc_name: str) -> bool:
        """
        Remove a processed document.
        """
        file_path = os.path.join(self.base_path, f"{doc_name}.json")

        if os.path.exists(file_path):
            os.remove(file_path)
            return True

        return False