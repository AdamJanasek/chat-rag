from typing import Dict, List, Optional

import httpx

from src.domain.llm import CompletionResponse, EmbeddingResponse
from src.services.base.ai_service import AIService


class OllamaService(AIService):
    def __init__(self, base_url: str = 'http://localhost:11434'):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)
        self.embedding_model = 'llama2'
        self.completion_model = 'llama2'

    async def create_embedding(self, text: str) -> EmbeddingResponse:
        response = await self.client.post(
            '/api/embeddings',
            json={
                'model': self.embedding_model,
                'prompt': text
            }
        )
        data = response.json()
        return EmbeddingResponse(
            embedding=data['embedding'],
            model=self.embedding_model
        )

    async def create_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None
    ) -> CompletionResponse:
        prompt = '\n'.join([f'{m['role']}: {m['content']}' for m in messages])

        response = await self.client.post(
            '/api/generate',
            json={
                'model': self.completion_model,
                'prompt': prompt,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )
        data = response.json()
        return CompletionResponse(
            content=data['response'],
            model=self.completion_model
        )

    def close(self) -> None:
        self.client.aclose()  # type: ignore
