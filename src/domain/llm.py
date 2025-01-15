from typing import Dict, List, Optional

from pydantic import BaseModel


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str


class CompletionResponse(BaseModel):
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
