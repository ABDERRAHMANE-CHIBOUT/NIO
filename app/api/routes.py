from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from app.pipeline.rag_pipeline import RAGPipeline
from app.pipeline.study_pipeline import StudyPipeline
from app.pipeline.json_pipeline import JSONPipeline

from app.dependencies.dependencies import (
    get_embedder,
    get_vector_store
)

from app.generation.factory import get_llm


router = APIRouter()


# =========================
# SCHEMAS
# =========================
class QueryRequest(BaseModel):
    question: str
    llm: str = "Qwen3-30B-A3B-Thinking"


class ChatRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    llm: str = "Qwen3-30B-A3B-Thinking"


class StudyRequest(BaseModel):
    topic: str
    llm: str = "default"
    doc_ids: Optional[List[str]] = None


# =========================
# SIMPLE JSON PIPELINE
# =========================
@router.post("/ask")
async def ask_question(request: QueryRequest):
    return JSONPipeline().run(request.question)


# =========================
# ✅ RAG CHAT (FIXED)
# =========================
@router.post("/chat")
def chat(request: ChatRequest):

    vector_store = get_vector_store()   # 🔥 always fresh
    rag = RAGPipeline(vector_store)     # 🔥 inject dependency

    response = rag.run(
        question=request.question,
        doc_ids=request.doc_ids,
        llm_name=request.llm
    )

    return response


# =========================
# STUDY PIPELINE
# =========================
@router.post("/study")
def study(request: StudyRequest):

    embedder = get_embedder()
    vector_store = get_vector_store()

    study = StudyPipeline(vector_store, embedder)
    llm = get_llm(request.llm)

    return study.run(
        topic=request.topic,
        doc_ids=request.doc_ids,
        llm=llm
    )

from fastapi import UploadFile, File, HTTPException
from app.utils.document_manager import DocumentManager
from app.pipeline.ingestion_pipeline import IngestionPipeline

doc_manager = DocumentManager()


# =========================
# 📥 UPLOAD DOCUMENT
# =========================
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # save file
    doc = doc_manager.save_document(file)

    # ingest into vector DB
    embedder = get_embedder()
    vector_store = get_vector_store()

    ingestion = IngestionPipeline(embedder, vector_store)
    num_chunks = ingestion.ingest(doc["path"], doc["doc_id"])

    # persist
    vector_store.save()

    return {
        "message": "Document uploaded successfully",
        "doc_id": doc["doc_id"],
        "chunks": num_chunks
    }


# =========================
# 📄 LIST DOCUMENTS
# =========================
@router.get("/documents")
def list_documents():
    return doc_manager.list_documents()


# =========================
# ❌ DELETE DOCUMENT
# =========================
@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):

    # delete from filesystem
    deleted = doc_manager.delete_document(doc_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    # delete from vector DB
    vector_store = get_vector_store()
    vector_store.delete({"doc_id": doc_id})
    vector_store.save()

    return {"message": "Document deleted successfully"}

@router.get("/models")
def list_models():
    return {
        "models": [
            "QWEN3-30B-A3B-THINKING",
            "GPT_OSS_120B"
        ]
    }
