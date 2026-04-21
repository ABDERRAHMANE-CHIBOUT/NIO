import json
from typing import List, Dict, Any, Optional
import numpy as np
import faiss


class LawsProcessor:
    """
    Single-module law engine:
    - loads raw laws.json
    - converts to RAG-friendly chunks
    - builds embeddings index (FAISS)
    - supports semantic search
    """

    def __init__(self, file_path="data/laws/laws.json", embedder=None):
        self.file_path = file_path
        self.embedder = embedder

        self.raw = self._load_raw()
        self.chunks = []
        self.index = None
        self.id_map = []

    # -------------------------
    # LOAD RAW JSON
    # -------------------------
    def _load_raw(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------
    # PUBLIC: BUILD EVERYTHING
    # -------------------------
    def build(self):
        """
        Main entry:
        - chunk laws
        - embed
        - build FAISS index
        """
        self.chunks = self._chunk_laws()

        if not self.embedder:
            raise ValueError("Embedder is required to build index")

        vectors = []
        self.id_map = []

        for chunk in self.chunks:
            vec = self.embedder.embed(chunk["text"])
            vectors.append(vec)
            self.id_map.append(chunk)

        vectors = np.array(vectors).astype("float32")

        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

        return self

    # -------------------------
    # PUBLIC: SEARCH
    # -------------------------
    def search(self, query_embedding, k=15):
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        query = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.id_map):
                results.append(self.id_map[idx])

        return results

    # -------------------------
    # CHUNKING CORE LOGIC
    # -------------------------
    def _chunk_laws(self) -> List[Dict[str, Any]]:
        chunks = []

        # -------------------------
        # INTRODUCTION
        # -------------------------
        intro = self.raw.get("introduction", {})
        if intro.get("text"):
            chunks.append(self._make_chunk(
                ctype="introduction",
                title="Introduction",
                text=intro["text"],
                path=["introduction"]
            ))

        # -------------------------
        # DEFINITIONS
        # -------------------------
        for d in self.raw.get("definitions", []):
            if not d.get("definition"):
                continue

            chunks.append(self._make_chunk(
                ctype="definition",
                title=d.get("term"),
                text=d.get("definition"),
                path=["definitions", d.get("term")]
            ))

        # -------------------------
        # TITLES → SECTIONS → ARTICLES
        # -------------------------
        for title in self.raw.get("titles", []):
            t_name = title.get("title_name")

            for section in title.get("sections", []):
                s_name = section.get("section_name")

                for article in section.get("articles", []):
                    text = article.get("content")

                    if not text:
                        continue

                    chunks.append(self._make_chunk(
                        ctype="article",
                        title=f"{article.get('article_number')} - {article.get('article_title') or ''}",
                        text=text,
                        path=[
                            "titles",
                            t_name,
                            s_name,
                            article.get("article_id")
                        ]
                    ))

                    # -------------------------
                    # SUBSECTIONS
                    # -------------------------
                    for sub in article.get("subsections", []):
                        if not sub.get("text"):
                            continue

                        chunks.append(self._make_chunk(
                            ctype="subsection",
                            title=sub.get("subtitle"),
                            text=sub.get("text"),
                            path=[
                                "titles",
                                t_name,
                                s_name,
                                article.get("article_id"),
                                sub.get("subtitle")
                            ]
                        ))

        return chunks

    # -------------------------
    # CHUNK FACTORY
    # -------------------------
    def _make_chunk(self, ctype, title, text, path):
        return {
            "id": self._make_id(path),
            "type": ctype,
            "title": title,
            "text": self._clean(text),
            "path": path
        }

    # -------------------------
    # CLEAN TEXT
    # -------------------------
    def _clean(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(text.split())

    # -------------------------
    # ID GENERATION
    # -------------------------
    def _make_id(self, path: List[str]) -> str:
        return "law::" + "::".join(str(p) for p in path if p)