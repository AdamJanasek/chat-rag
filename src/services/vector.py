import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.grpc import ScoredPoint
from qdrant_client.http import models

from src.services.base.ai_service import AIService


class VectorService:
    def __init__(self, ai_service: AIService, qdrant_client: QdrantClient):
        self.client = qdrant_client
        self.ai_service = ai_service

    async def ensure_collection(self, name: str) -> None:
        collections = self.client.get_collections()
        if not any(collection.name == name for collection in collections.collections):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            )

    async def initialize_collection_with_data(self, name: str, points: List[Dict[str, Any]]) -> None:
        await self.ensure_collection(name)
        await self.add_points(name, points)

    async def create_embedding(self, text: str) -> List[float]:
        response = await self.ai_service.create_embedding(text)
        return response.embedding

    async def add_points(self, collection_name: str, points: List[Dict[str, Any]]) -> None:
        points_to_upsert = []
        for point in points:
            embedding = await self.create_embedding(point['text'])
            point_id = point.get('id', str(uuid.uuid4()))

            point_struct = models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'text': point['text'],
                    'filename': point['payload']['filename'],
                    'chunk_index': point['payload']['chunk_index'],
                    'headers': point['payload']['headers'],
                    'urls': point['payload']['urls'],
                    'images': point['payload']['images'],
                    'tokens': point['payload']['tokens']
                }
            )
            points_to_upsert.append(point_struct)

        points_file = Path('storage/points')
        points_file.mkdir(parents=True, exist_ok=True)
        points_file = points_file / 'points.json'

        points_file.write_text(json.dumps(
            [p.dict() for p in points_to_upsert],
            indent=2,
            default=str,
        ))

        self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points_to_upsert
        )

    async def perform_search(
            self,
            collection_name: str,
            query: str,
            filter_: Optional[Dict[str, Any]] = None,
            limit: int = 5,
            rerank: bool = True
    ) -> List[ScoredPoint]:
        query_embedding = await self.create_embedding(query)

        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit if not rerank else limit * 2,
            query_filter=filter_,
            with_payload=True
        )

        if not rerank:
            return search_results

        reranked_results = []
        for result in search_results:
            system_content = '''
            You are a helpful assistant that determines if a given text is relevant to a query.
            Respond with a number between 0 and 1, where 1 is highly relevant and 0 is not relevant at all.
            '''
            relevance_check = await self.ai_service.create_completion(
                messages=[
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': f'Query: {query}\nText: {result.payload['text']}'}  # type: ignore
                ]
            )

            relevance_score = float(relevance_check.content)
            reranked_results.append({
                **result.dict(),
                'relevance_score': relevance_score,
                'combined_score': (result.score + relevance_score) / 2
            })

        reranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return reranked_results[:limit]
