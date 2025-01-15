from asyncio import Protocol
from typing import Any, Dict, List, Optional

from src.console.domain import QueryResult


class MessageRepository(Protocol):
    def add_message(self, role: str, content: str) -> None:
        return NotImplemented

    def get_context(self) -> List[Dict[str, str]]:
        return NotImplemented


class QueryService(Protocol):
    async def query(
            self,
            query: str,
            chat_history: List[Dict[str, str]],
            top_k: int,
            rerank: bool,
            filter_: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        return NotImplemented


class UserInterface(Protocol):
    def display_welcome(self) -> None:
        return NotImplemented

    def display_response(self, result: QueryResult) -> None:
        return NotImplemented

    def display_error(self, error: Exception) -> None:
        return NotImplemented

    def get_user_input(self) -> str:
        return NotImplemented
