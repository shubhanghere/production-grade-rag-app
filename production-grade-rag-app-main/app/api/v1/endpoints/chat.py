from fastapi import APIRouter, HTTPException

from app.core.rag import query_rag_with_context, source_citations
from app.schemas.models import ChatRequest, ChatResponse, Citation

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    result = await query_rag_with_context(request.question)
    citations = [Citation(**c) for c in source_citations(result.documents)]
    return ChatResponse(answer=result.answer, citations=citations)
