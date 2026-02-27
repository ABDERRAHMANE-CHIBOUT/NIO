import os

def load_documents(folder_path: str):
    documents = []

    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            documents.append(f.read())

    return documents
