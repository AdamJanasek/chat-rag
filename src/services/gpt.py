import os
from typing import Dict, List, Optional

import openai

from src.domain.llm import CompletionResponse, EmbeddingResponse
from src.services.base.ai_service import AIService


class OpenAIService(AIService):
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.embedding_model = 'text-embedding-ada-002'
        self.completion_model = 'gpt-4o-mini'

    async def create_embedding(self, text: str) -> EmbeddingResponse:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return EmbeddingResponse(
            embedding=response.data[0].embedding,
            model=self.embedding_model
        )

    async def create_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None
    ) -> CompletionResponse:
        response = self.client.chat.completions.create(
            model=self.completion_model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens
        )

        usage = response.usage.model_dump() if response.usage else {}
        usage = {
            k: v for k, v in usage.items()
            if isinstance(v, (int, float))
        }
        return CompletionResponse(
            content=response.choices[0].message.content,
            model=self.completion_model,
            usage=usage,
        )

    def close(self) -> None:
        pass
