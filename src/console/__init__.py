from typing import Any, Dict, List, Optional

import httpx
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.console.domain import QueryResult, Source
from src.console.interfaces import QueryService


class HttpQueryService(QueryService):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def query(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        top_k: int,
        rerank: bool,
        filter_: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        url = f'{self.base_url}/query'
        params = {
            'query': query,
            'top_k': top_k,
            'rerank': rerank,
            'chat_history': chat_history
        }
        if filter_:
            params['filter_'] = filter_

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with Progress(  # type: ignore
                SpinnerColumn(),
                TextColumn('[progress.description]{task.description}'),
                transient=True,
            ) as progress:
                progress.add_task(description='Thinking...', total=None)
                response = await client.post(url, json=params)
                response.raise_for_status()
                data = response.json()
                return QueryResult(
                    answer=data['answer'],
                    sources=[Source(**source) for source in data.get('sources', [])]
                )
