import os
import uuid
from typing import List

DATA_DIR = "data/raw"

class DocumentManager:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    def save_document(self, file) -> dict:
        doc_id = str(uuid.uuid4())
        file_path = os.path.join(DATA_DIR, f"{doc_id}_{file.filename}")

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "path": file_path
        }

    def list_documents(self) -> List[dict]:
        docs = []
        for file in os.listdir(DATA_DIR):
            doc_id = file.split("_")[0]
            docs.append({
                "doc_id": doc_id,
                "filename": file
            })
        return docs
    
    def get_document(self, doc_id: str):
        for file in os.listdir(DATA_DIR):
            if file.startswith(doc_id):
                return {
                    "doc_id": doc_id,
                    "filename": file,
                    "path": os.path.join(DATA_DIR, file)  # 🔥 FIX
                }
        return None

    def delete_document(self, doc_id: str):
        for file in os.listdir(DATA_DIR):
            if file.startswith(doc_id):
                os.remove(os.path.join(DATA_DIR, file))
                return True
        return False