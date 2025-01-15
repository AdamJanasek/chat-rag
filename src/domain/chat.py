from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    rerank: bool = True
    filter_: Optional[Dict[str, Any]] = None
    temperature: float = 0.7
    chat_history: Optional[List[Message]] = None
