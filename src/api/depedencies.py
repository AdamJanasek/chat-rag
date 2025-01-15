from fastapi import Request

from src.services.vector import VectorService


def get_vector_service(request: Request) -> VectorService:
    return request.app.container.get_service('vector')
