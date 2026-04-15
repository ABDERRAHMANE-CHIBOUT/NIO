# NIO/main.py (or NIO/app/main.py — wherever your main.py is)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(title='NIO — Naftal Intelligence Optimizer')

# =============================================================
# CORS — allows React frontend (Vite port 5173) to call this
# =============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get('/')
def root():
    return {'message': 'NIO RAG is working'}

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'NIO RAG Backend'}
