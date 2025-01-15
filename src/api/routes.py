from fastapi import APIRouter, Depends, UploadFile

from src.api.depedencies import get_vector_service
from src.domain.chat import QueryRequest
from src.domain.response import QueryResponse, UploadResponse
from src.services.document import DocumentService
from src.services.query import QueryService
from src.services.vector import VectorService

router = APIRouter()


@router.post('/upload')
async def upload_document(
    file: UploadFile,
    vector_service: VectorService = Depends(get_vector_service)
) -> UploadResponse:
    document_service = DocumentService(vector_service)
    return await document_service.process_document(file)


@router.post('/query')
async def query_documents(
    request: QueryRequest,
    vector_service: VectorService = Depends(get_vector_service)
) -> QueryResponse:
    query_service = QueryService(vector_service)
    return await query_service.process_query(request)
