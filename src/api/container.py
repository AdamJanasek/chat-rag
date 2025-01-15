from typing import Any, Dict

from qdrant_client import QdrantClient

from src.services.vector import VectorService
from src.settings import Settings
from src.utils.utils import create_ai_service


class Container:
    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}

    def init_resources(self, settings: Settings) -> None:
        self._services['ai'] = create_ai_service(settings.AI_PROVIDER)
        self._services['qdrant'] = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self._services['vector'] = VectorService(
            self._services['ai'],
            self._services['qdrant']
        )

    async def cleanup(self) -> None:
        if 'ai' in self._services:
            self._services['ai'].close()
        if 'qdrant' in self._services:
            self._services['qdrant'].close()
        self._services.clear()

    def get_service(self, name: str) -> Any:
        return self._services.get(name)
