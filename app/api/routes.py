from fastapi import APIRouter
from pydantic import BaseModel
from app.pipeline.json_pipeline import JSONPipeline


router = APIRouter()
#pipeline = RAGPipeline()
pipeline = JSONPipeline()

class QueryRequest(BaseModel):
    question: str

@router.post("/ask")
def ask_question(request: QueryRequest):
    response = pipeline.run(request.question)
    return {"answer": response}
