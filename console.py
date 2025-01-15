import asyncio

import typer

from src.console.chat import ChatService
from src.console.message import InMemoryMessageRepository
from src.console.query import HttpQueryService
from src.console.rich_console import RichConsoleInterface


def create_app() -> typer.Typer:
    app = typer.Typer()

    @app.command()
    def chat(
            top_k: int = typer.Option(3, "--top-k", "-k", help="Number of top results to consider"),
            rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Whether to rerank results"),
            base_url: str = typer.Option("http://app:8000", "--base-url", help="Base URL for the API")
    ) -> None:
        message_repository = InMemoryMessageRepository()
        query_service = HttpQueryService(base_url=base_url)
        user_interface = RichConsoleInterface()

        chat_service = ChatService(message_repository, query_service, user_interface)
        asyncio.run(chat_service.start_chat(top_k, rerank))

    return app


if __name__ == "__main__":
    app = create_app()
    app()
