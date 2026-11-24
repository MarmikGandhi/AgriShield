"""
chat.py
=======
Kisan Bot AI chat interface endpoint with multilingual agronomic RAG.
"""

from fastapi import APIRouter
from app.schemas.schemas import ChatRequest, ChatResponse
from app.services.kisan_bot_service import generate_kisan_bot_reply

router = APIRouter(prefix="/chat", tags=["Kisan Bot"])

@router.post("", response_model=ChatResponse)
def chat_with_bot(req: ChatRequest):
    response = generate_kisan_bot_reply(
        message=req.message,
        crop=req.crop,
        language=req.language
    )
    return response
