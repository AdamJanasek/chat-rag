from typing import Any, Dict, Optional

from pydantic import BaseModel


class SearchResult(BaseModel):
    score: float
    relevance_score: Optional[float]
    combined_score: Optional[float]
    payload: Dict[str, Any]
