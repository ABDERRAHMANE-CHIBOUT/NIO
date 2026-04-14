from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents


class IngestionPipeline:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self, file_path: str, doc_id: str):
        raw_docs = load_documents(file_path)
        split_docs = split_documents(raw_docs)

        texts = [doc.page_content for doc in split_docs]

        embeddings = self.embedder.embed(texts)

        # attach doc_id to each chunk
        metadatas = [
                    {
                        "doc_id": doc_id,
                        "chunk_id": i
                    }
                    for i, _ in enumerate(texts)
                ]

        self.vector_store.add(
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas
        )

        return len(texts)