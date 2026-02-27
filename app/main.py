from fastapi import FastAPI
from app.api.routes import routes

app = FastAPI(title='RAG system')
app.include_router(routes)

@app.get('/')
def rout():
    return{'message' : 'RAG is working'}