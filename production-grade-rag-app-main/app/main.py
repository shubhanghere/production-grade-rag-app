from dotenv import load_dotenv
load_dotenv(override=True)  # Always prefer .env over any pre-existing shell variables.

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as api_router

app = FastAPI(
    title="RAG LLM App",
    description="A LangChain-powered RAG application using hybrid retrieval (dense + BM25) with reranking.",
    version="0.1.0",
)

app.include_router(api_router, prefix="/api/v1")

# Serve the SPA last so API routes take precedence.
# html=True means index.html is served for directory requests (including /).
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
