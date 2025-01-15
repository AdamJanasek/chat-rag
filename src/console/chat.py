from src.console.interfaces import MessageRepository, QueryService, UserInterface


class ChatService:
    def __init__(
        self,
        message_repository: MessageRepository,
        query_service: QueryService,
        user_interface: UserInterface
    ):
        self.message_repository = message_repository
        self.query_service = query_service
        self.user_interface = user_interface

    async def handle_query(self, query: str, top_k: int, rerank: bool) -> None:
        try:
            self.message_repository.add_message('user', query)
            result = await self.query_service.query(
                query=query,
                chat_history=self.message_repository.get_context(),
                top_k=top_k,
                rerank=rerank
            )
            self.message_repository.add_message('assistant', result.answer)
            self.user_interface.display_response(result)
        except Exception as e:
            self.user_interface.display_error(e)

    async def start_chat(self, top_k: int, rerank: bool) -> None:
        self.user_interface.display_welcome()

        while True:
            query = self.user_interface.get_user_input()

            if query.lower() in ('exit', 'quit'):
                self.user_interface.console.print('\n[yellow]Goodbye! 👋[/yellow]')  # type: ignore
                break

            await self.handle_query(query, top_k, rerank)
