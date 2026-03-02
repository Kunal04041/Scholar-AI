from pydantic import BaseModel
from typing import Optional, List


class IngestRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    source_name: Optional[str] = "unknown"


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5
    evaluate: Optional[bool] = False


class SourceDoc(BaseModel):
    content: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    session_id: str
    evaluation: Optional[dict] = None


class IngestResponse(BaseModel):
    status: str
    chunks_stored: int
    source: str
