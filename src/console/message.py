from typing import Dict, List

from src.console.domain import Message
from src.console.interfaces import MessageRepository


class InMemoryMessageRepository(MessageRepository):
    def __init__(self) -> None:
        self._messages: List[Message] = []

    def add_message(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))

    def get_context(self) -> List[Dict[str, str]]:
        return [{'role': msg.role, 'content': msg.content} for msg in self._messages]
