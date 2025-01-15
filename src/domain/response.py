from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Source(BaseModel):
    filename: str
    chunk_index: Optional[int]
    relevance_score: Optional[float]
    combined_score: Optional[float]
    vector_score: float
    headers: Dict[str, Any]
    urls: List[str]


class QueryMetadata(BaseModel):
    reranked: bool
    search_time_ms: float
    completion_time_ms: float
    total_tokens: Optional[int]
    timestamp: datetime
    history_length: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    metadata: QueryMetadata


class UploadResponse(BaseModel):
    message: str
