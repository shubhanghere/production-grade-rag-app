from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    title: str
    url: str
    section_id: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []


class IngestRequest(BaseModel):
    text: str
    source: str = "uploaded"


class IngestResponse(BaseModel):
    section_id: str
    message: str
