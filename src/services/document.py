import re
import uuid
from typing import Any, Dict, List

from fastapi import HTTPException, UploadFile

from src.domain.response import UploadResponse
from src.services.vector import VectorService
from src.splitters.text_splitter import TextSplitter

COLLECTION_NAME = 'ai_course_docs'


class DocumentService:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.text_splitter = TextSplitter()

    async def process_document(self, file: UploadFile) -> UploadResponse:
        if not file.filename.endswith('.md'):  # type: ignore
            raise HTTPException(
                status_code=400,
                detail='Only markdown files are supported'
            )

        content = await file.read()
        text = content.decode('utf-8')
        cleaned_text = self._clean_markdown_links(text)

        chunks = self.text_splitter.split(cleaned_text, limit=1000)
        points = self._create_points(chunks, text, file.filename)  # type: ignore

        await self.vector_service.add_points(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        return UploadResponse(message='Document processed successfully')

    def _clean_markdown_links(self, text: str) -> str:
        return re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', text)

    def _create_points(self, chunks: List[Any], original_text: str, filename: str) -> List[Dict[str, Any]]:
        points = []
        for index, chunk in enumerate(chunks):
            original_chunk_text = chunk.text
            if hasattr(chunk, 'start') and hasattr(chunk, 'end'):
                original_chunk_text = original_text[chunk.start:chunk.end]

            points.append({
                'id': str(uuid.uuid4()),
                'text': chunk.text,
                'payload': {
                    'filename': filename,
                    'chunk_index': index,
                    'headers': chunk.metadata.headers,
                    'urls': chunk.metadata.urls,
                    'images': chunk.metadata.images,
                    'tokens': chunk.metadata.tokens,
                    'original_text': original_chunk_text,
                },
            })
        return points
