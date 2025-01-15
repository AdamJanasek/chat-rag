from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Source:
    filename: str
    relevance_score: Optional[float] = None
    vector_score: Optional[float] = None
    combined_score: Optional[float] = None
    chunk_index: Optional[int] = None
    headers: Optional[Dict[str, str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Source':
        filename = data.pop('filename')
        relevance_score = data.pop('relevance_score', None)
        vector_score = data.pop('vector_score', None)
        combined_score = data.pop('combined_score', None)
        chunk_index = data.pop('chunk_index', None)
        headers = data.pop('headers', None)
        return cls(
            filename=filename,
            relevance_score=relevance_score,
            vector_score=vector_score,
            combined_score=combined_score,
            chunk_index=chunk_index,
            headers=headers
        )


@dataclass
class QueryResult:
    answer: str
    sources: List[Source]
