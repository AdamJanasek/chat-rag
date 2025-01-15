import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from src.domain.chat import QueryRequest
from src.domain.response import QueryMetadata, QueryResponse, Source
from src.services.vector import VectorService
from src.utils.utils import format_search_result


class QueryService:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        self._validate_request(request)

        search_results, search_time = await self._perform_search(request)
        if not search_results:
            return self._create_empty_response(search_time, request.rerank)

        context = self._create_context(search_results)
        messages = self._prepare_messages(context, request)

        completion, completion_time = await self._get_completion(messages, request)

        return self._create_response(
            completion,
            search_results,
            search_time,
            completion_time,
            request
        )

    def _validate_request(self, request: QueryRequest) -> None:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail='Query cannot be empty')
        if request.top_k < 1:
            raise HTTPException(status_code=400, detail='top_k must be at least 1')

    async def _perform_search(self, request: QueryRequest) -> Tuple[List[Dict[str, Any]], float]:
        start_time = time.time()
        results = await self.vector_service.perform_search(
            collection_name='ai_course_docs',
            query=request.query,
            filter_=request.filter_,
            limit=request.top_k,
            rerank=request.rerank
        )
        search_time = time.time() - start_time
        return results, search_time

    def _create_context(self, search_results: List[Dict[str, Any]]) -> str:
        context_parts = [
            format_search_result(result)
            for result in search_results
        ]
        return '\n\n'.join(context_parts)

    def _prepare_messages(self, context: str, request: QueryRequest) -> List[Dict[str, str]]:
        system_content = '''
        You are a helpful assistant for an AI course. Follow these guidelines:
        - Use the provided context to answer questions accurately
        - When citing information, use the source markers [filename] provided in the context
        - If you're unsure or the context doesn't contain relevant information, say so
        - Keep answers concise but comprehensive
        - Format your response using markdown when appropriate
        - Maintain conversation coherence by considering the chat history
        - If referring to previous exchanges, be explicit about what was discussed
        '''

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'system', 'content': f'Here is relevant context for the current question:\n{context}'}
        ]

        if request.chat_history:
            recent_history = request.chat_history[-5:]
            messages.extend([
                {'role': msg.role, 'content': msg.content}
                for msg in recent_history
            ])

        messages.append({
            'role': 'user',
            'content': f'''
                Question: {request.query}\n
                Please provide a well-structured answer using the provided context.
            '''
        })

        return messages

    async def _get_completion(self, messages: List[Dict[str, str]], request: QueryRequest) -> Tuple[Any, float]:
        start_time = time.time()
        completion = await self.vector_service.ai_service.create_completion(
            messages=messages,
            temperature=request.temperature,
        )
        completion_time = time.time() - start_time
        return completion, completion_time

    def _create_empty_response(self, search_time: float, reranked: bool) -> QueryResponse:
        return QueryResponse(
            answer="I couldn't find any relevant information to answer your question.",
            sources=[],
            metadata=QueryMetadata(
                reranked=reranked,
                search_time_ms=round(search_time * 1000, 2),
                completion_time_ms=0,
                total_tokens=None,
                timestamp=datetime.utcnow(),
                history_length=0
            )
        )

    def _create_response(
            self,
            completion: Any,
            search_results: List[Dict[str, Any]],
            search_time: float,
            completion_time: float,
            request: QueryRequest
    ) -> QueryResponse:
        total_tokens = self._get_total_tokens(completion)

        return QueryResponse(
            answer=completion.content if hasattr(completion, 'content') else completion.get('content', ''),
            sources=[
                Source(
                    filename=result['payload'].get('filename', 'unknown'),
                    chunk_index=result['payload'].get('chunk_index'),
                    relevance_score=result.get('relevance_score'),
                    combined_score=result.get('combined_score'),
                    vector_score=result['score'],
                    headers=result['payload'].get('headers', {}),
                    urls=result['payload'].get('urls', []),
                )
                for result in search_results
            ],
            metadata=QueryMetadata(
                reranked=request.rerank,
                search_time_ms=round(search_time * 1000, 2),
                completion_time_ms=round(completion_time * 1000, 2),
                total_tokens=total_tokens,
                timestamp=datetime.utcnow(),
                history_length=len(request.chat_history) if request.chat_history else 0
            )
        )

    def _get_total_tokens(self, completion: Any) -> Optional[int]:
        if isinstance(completion, dict):
            return completion.get('usage', {}).get('total_tokens')
        elif hasattr(completion, 'usage'):
            return getattr(completion.usage, 'total_tokens', None)
        return None
