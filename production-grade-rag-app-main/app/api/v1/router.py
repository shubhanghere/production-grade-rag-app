from fastapi import APIRouter

from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.documents import router as documents_router

router = APIRouter()
router.include_router(chat_router, tags=["Chat"])
router.include_router(documents_router, tags=["Documents"])
