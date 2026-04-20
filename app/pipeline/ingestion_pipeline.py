# app/pipeline/ingestion_pipeline.py

import os
from pathlib import Path
from app.ingestion.loader import load_file
from app.ingestion.splitter import split_documents
from langchain_core.documents import Document
import numpy as np


class IngestionPipeline:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self, file_path: str, doc_id: str):
        # ── Load single file (not a folder) ──────────────────────────────
        content = load_file(file_path)
        file_name = Path(file_path).name

        # Build Document object(s) from the content
        raw_docs = []
        if isinstance(content, list):
            # PDF returns list of pages
            for page in content:
                if page["text"].strip():
                    raw_docs.append(
                        Document(
                            page_content=page["text"],
                            metadata={
                                "source": file_name,
                                "path": file_path,
                                "page": page["page"],
                                "doc_id": doc_id,
                            }
                        )
                    )
        else:
            # TXT / DOCX / JSON returns a string
            if content.strip():
                raw_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "path": file_path,
                            "doc_id": doc_id,
                        }
                    )
                )

        if not raw_docs:
            print(f"[WARN] No content extracted from {file_path}")
            return 0

        # ── Split into chunks ─────────────────────────────────────────────
        split_docs = split_documents(raw_docs)

        if not split_docs:
            print(f"[WARN] No chunks generated from {file_path}")
            return 0

        texts = [doc.page_content for doc in split_docs]

        # ── Embed ─────────────────────────────────────────────────────────
        embeddings = self.embedder.embed(texts)
        embeddings = np.array(embeddings, dtype="float32")

        if embeddings.ndim != 2:
            print(f"[WARN] Bad embedding shape {embeddings.shape}, skipping")
            return 0

        # ── Attach doc_id to metadata ─────────────────────────────────────
        metadatas = [
            {
                "doc_id": doc_id,
                "chunk_id": i,
                "source": file_name,
            }
            for i, _ in enumerate(texts)
        ]

        self.vector_store.add(
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas
        )

        print(f"✅ Ingested {len(texts)} chunks from {file_name}")
        return len(texts)
