# app/api/routes.py
import uuid
from typing import Optional, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.pipeline.rag_pipeline import RAGPipeline
from app.pipeline.study_pipeline import StudyPipeline
from app.pipeline.json_pipeline import JSONPipeline
from app.dependencies.dependencies import get_embedder, get_vector_store
from app.generation.factory import get_llm
from app.retrieval.retriever import Retriever
from app.utils.document_manager import DocumentManager
from app.pipeline.ingestion_pipeline import IngestionPipeline

router = APIRouter()
doc_manager = DocumentManager()

# =============================================================
# IN-MEMORY CONVERSATION STORE
# =============================================================
conversations: dict = {}


# =============================================================
# SCHEMAS
# =============================================================
class QueryRequest(BaseModel):
    question: str
    llm: str = "Qwen3-30B-A3B-Thinking"


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    doc_ids: Optional[list] = []
    llm: str = "Qwen3-30B-A3B-Thinking"


class StudyRequest(BaseModel):
    conversation_id: Optional[str] = None
    topic: str
    doc_ids: Optional[list] = []
    llm: str = "QWEN3-30B-A3B-THINKING"


# =============================================================
# PROVIDER MAPPER
# Maps frontend model name → exact .env key prefix
# =============================================================
def map_llm_provider(llm_name: str) -> str:
    mapping = {
        "Qwen3-30B-A3B-Thinking": "QWEN3-30B-A3B-THINKING",
        "GPT_OSS_120B":           "GPT_OSS_120B",
        "default":                "QWEN3-30B-A3B-THINKING",
    }
    return mapping.get(llm_name, "QWEN3-30B-A3B-THINKING")


# =============================================================
# HEALTH
# =============================================================
@router.get("/health")
def health():
    return {"status": "ok", "service": "NIO RAG Backend"}


# =============================================================
# MODELS
# =============================================================
@router.get("/models")
def list_models():
    return {
        "models": [
            "Qwen3-30B-A3B-Thinking",
            "GPT_OSS_120B",
        ]
    }


# =============================================================
# CONVERSATIONS — create
# =============================================================
@router.post("/conversations")
def create_conversation():
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = {
        "title": "Nouvelle discussion",
        "messages": []
    }
    return {
        "conversation_id": conversation_id,
        "title": "Nouvelle discussion"
    }


# =============================================================
# CONVERSATIONS — list all
# =============================================================
@router.get("/conversations")
def get_conversations():
    return [
        {"conversation_id": cid, "title": data["title"]}
        for cid, data in conversations.items()
    ]


# =============================================================
# CONVERSATIONS — get one
# =============================================================
@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    return conversations[conversation_id]["messages"]


# =============================================================
# CONVERSATIONS — delete
# =============================================================
@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    del conversations[conversation_id]
    return {"message": "Conversation supprimée"}


# =============================================================
# SIMPLE JSON PIPELINE
# =============================================================
@router.post("/ask")
async def ask_question(request: QueryRequest):
    return JSONPipeline().run(request.question)


# =============================================================
# RAG CHAT
# =============================================================
@router.post("/chat")
async def chat(
    request: ChatRequest,
    vector_store=Depends(get_vector_store)
):
    # Auto-create conversation if not provided or not found
    conv_id = request.conversation_id
    if not conv_id or conv_id not in conversations:
        conv_id = str(uuid.uuid4())
        conversations[conv_id] = {
            "title": request.question[:40],
            "messages": []
        }

    # Save user message
    conversations[conv_id]["messages"].append({
        "role": "user",
        "content": request.question
    })

    # Run RAG pipeline
    result = {}
    try:
        provider = map_llm_provider(request.llm)        # ✅ maps to .env key
        pipeline = RAGPipeline(vector_store=vector_store)
        result = pipeline.run(
            question=request.question,
            doc_ids=request.doc_ids or [],
            llm_name=provider                           # ✅ exact .env key
        )
        answer = result.get("answer") or result.get("response") or str(result)
    except Exception as e:
        answer = f"Erreur RAG : {str(e)}"

    # Save assistant message
    conversations[conv_id]["messages"].append({
        "role": "assistant",
        "content": answer
    })

    # Auto-title from first exchange
    if len(conversations[conv_id]["messages"]) == 2:
        conversations[conv_id]["title"] = request.question[:50]

    return {
        "conversation_id": conv_id,
        "answer": answer,
        "sources": result.get("sources", []) if isinstance(result, dict) else []
    }


# =============================================================
# STUDY / ANALYSE
# =============================================================
@router.post("/study")
async def study(request: StudyRequest):

    conv_id = request.conversation_id

    if not conv_id or conv_id not in conversations:
        conv_id = str(uuid.uuid4())
        conversations[conv_id] = {
            "title": f"Analyse : {request.topic[:30]}",
            "messages": []
        }

    conversations[conv_id]["messages"].append({
        "role": "user",
        "content": f"[Study] {request.topic}"
    })

    try:
        # ❌ NO retriever, NO llm injection
        pipeline = StudyPipeline()

        result = pipeline.run()

        answer = result.get("study", str(result))

    except Exception as e:
        answer = f"Erreur analyse : {str(e)}"

    conversations[conv_id]["messages"].append({
        "role": "assistant",
        "content": answer
    })

    return {
        "conversation_id": conv_id,
        "answer": answer,
        "json_path": result.get("json_path"),
        "mode": result.get("mode")
    }


# =============================================================
# UPLOAD DOCUMENT
# =============================================================
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    doc = doc_manager.save_document(file)

    embedder     = get_embedder()
    vector_store = get_vector_store()

    ingestion  = IngestionPipeline(embedder, vector_store)
    num_chunks = ingestion.ingest(doc["path"], doc["doc_id"])

  

    return {
        "message": "Document uploaded successfully",
        "doc_id": doc["doc_id"],
        "chunks": num_chunks
    }


# =============================================================
# LIST DOCUMENTS
# =============================================================
@router.get("/documents")
def list_documents():
    return doc_manager.list_documents()


# =============================================================
# DELETE DOCUMENT
# =============================================================
@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    deleted = doc_manager.delete_document(doc_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    vector_store = get_vector_store()
    vector_store.delete({"doc_id": doc_id})
    vector_store.save()

    return {"message": "Document deleted successfully"}
