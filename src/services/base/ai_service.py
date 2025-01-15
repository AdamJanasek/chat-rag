from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.domain.llm import CompletionResponse, EmbeddingResponse


class AIService(ABC):
    @abstractmethod
    async def create_embedding(self, text: str) -> EmbeddingResponse:
        pass

    @abstractmethod
    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> CompletionResponse:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
