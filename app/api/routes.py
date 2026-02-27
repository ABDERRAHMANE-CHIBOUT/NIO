from fastapi import APIRouter
from pydantic import BaseModel
from app.pipeline.rag_pipeline import RAGPipeline

router = APIRouter()
pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    question: str

@router.post("/ask")
def ask_question(request: QueryRequest):
    response = pipeline.run(request.question)
    return {"answer": response}
