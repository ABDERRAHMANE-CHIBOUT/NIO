# app/api/routes.py
import uuid
from typing import Optional, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.pipeline.rag_pipeline import RAGPipeline
from app.pipeline.study_pipeline import StudyPipeline
from app.dependencies.dependencies import get_embedder, get_vector_store, get_laws_processor
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
    llm: str


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    doc_ids: Optional[list] = []
    llm: str


class StudyRequest(BaseModel):
    conversation_id: Optional[str] = None
    topic: str
    doc_ids: Optional[list] = []
    llm: str


# =============================================================
# PROVIDER MAPPER
# Maps frontend model name → exact .env key prefix
# =============================================================
def map_llm_provider(llm_name: str) -> str:
    llm_name = llm_name.strip().lower()

    mapping = {
        "qwen3-30b-a3b-thinking": "QWEN3-30B-A3B-THINKING",
        "mistralai/voxtral-mini-4b-realtime-2602": "MISTRALAI_VOXTRAL_MINI_4B_REALTIME_2602",
        "gpt-oss-120b": "GPT_OSS_120B",
        "google/gemma-4-31b": "GOOGLE_GEMMA_4_31B"
    }

    provider = mapping.get(llm_name)

    if not provider:
        print(f"⚠️ MODEL NOT FOUND ({llm_name}) → FALLBACK TO QWEN")
        return "QWEN3-30B-A3B-THINKING"

    return provider
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
            "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "gpt-oss-120b",
            "google/gemma-4-31B"
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
# RAG CHAT
# =============================================================
@router.post("/chat")
async def chat(
    request: ChatRequest,
    vector_store=Depends(get_vector_store),
    embedder=Depends(get_embedder),
    laws_processor=Depends(get_laws_processor)
):

    conv_id = request.conversation_id

    # -------------------------
    # CREATE NEW CONVERSATION
    # -------------------------
    if not conv_id or conv_id not in conversations:
        conv_id = str(uuid.uuid4())
        conversations[conv_id] = {
            "title": request.question[:40],
            "messages": []
        }

    conversations[conv_id]["messages"].append({
        "role": "user",
        "content": request.question
    })

    provider = map_llm_provider(request.llm)
    print("RAW REQUEST MODEL:", request.llm)

    result = {}
    answer = ""

    try:
        llm = get_llm(provider)

        # -------------------------
        # PIPELINE (NOW FULLY WIRED)
        # -------------------------
        pipeline = RAGPipeline(
            llm=llm,
            vector_store=vector_store,
            embedder=embedder,
            laws_processor=laws_processor   # 🔥 FIXED
        )

        # -------------------------
        # MODE CONTROL
        # -------------------------
        if not request.doc_ids:
            result = pipeline.run(
                question=request.question,
                doc_ids=None,
                llm_name=provider,
                mode="laws"
            )
        else:
            result = pipeline.run(
                question=request.question,
                doc_ids=request.doc_ids,
                llm_name=provider,
                mode="rag"
            )

        answer = result.get("answer") or str(result)

    except Exception as e:
        answer = f"Erreur RAG : {str(e)}"

    conversations[conv_id]["messages"].append({
        "role": "assistant",
        "content": answer
    })

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

    result = None
    answer = None

    try:
        pipeline = StudyPipeline()

        result = pipeline.run(
            domain=request.topic  # optional but better
        )

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
        "json_path": result.get("json_path") if result else None,
        "mode": result.get("mode") if result else None
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
