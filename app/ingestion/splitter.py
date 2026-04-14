from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=[
            "\n\n",   # paragraphs (highest priority)
            "\n",     # lines
            ". ",     # sentences
            " "       # fallback
        ]
    )

    chunks = splitter.split_documents(documents)

    # ---------------------------
    # Metadata enrichment
    # ---------------------------
    for i, chunk in enumerate(chunks):
        if not chunk.metadata:
            chunk.metadata = {}

        chunk.metadata["chunk_id"] = i

        text = chunk.page_content.lower()

        # 🔥 simple structure detection
        if "article" in text:
            chunk.metadata["type"] = "article"
        elif "section" in text:
            chunk.metadata["type"] = "section"
        elif "chapter" in text:
            chunk.metadata["type"] = "chapter"
        else:
            chunk.metadata["type"] = "text"

    return chunks