from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title='RAG system')
app.include_router(router)

@app.get('/')
def rout():
    return{'message' : 'RAG is working'}