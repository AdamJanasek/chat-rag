from typing import List

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.console import QueryResult, Source
from src.console.interfaces import UserInterface


class RichConsoleInterface(UserInterface):
    def __init__(self) -> None:
        self.console = Console()

    def display_welcome(self) -> None:
        self.console.print(Panel.fit(
            '🤖 [bold blue]AI Course Assistant[/bold blue]\n'
            '[dim]Type your questions and press Enter. Type "exit" to quit.[/dim]',
            border_style='blue'
        ))

    def display_response(self, result: QueryResult) -> None:
        self.console.print('\n[bold blue]Assistant[/bold blue]')
        self.console.print(Panel(
            str(result.answer),
            border_style='blue',
            padding=(1, 2)
        ))

        if result.sources:
            self.console.print('\n[bold]Reference Sources:[/bold]')
            self._display_sources(result.sources)

    def _display_sources(self, sources: List[Source]) -> None:
        table = Table(title='Sources', show_header=True, header_style='bold magenta')
        table.add_column('Filename', style='cyan')
        table.add_column('Relevance Score', justify='right', style='green')
        table.add_column('Vector Score', justify='right', style='yellow')
        table.add_column('Combined Score', justify='right', style='blue')

        for source in sources:
            table.add_row(
                source.filename,
                f'{source.relevance_score:.4f}' if source.relevance_score is not None else 'N/A',
                f'{source.vector_score:.4f}' if source.vector_score is not None else 'N/A',
                f'{source.combined_score:.4f}' if source.combined_score is not None else 'N/A'
            )

        self.console.print(table)

    def display_error(self, error: Exception) -> None:
        if isinstance(error, httpx.HTTPStatusError):
            self.console.print(f'\n[red]Error: HTTP {error.response.status_code}[/red]')
            self.console.print(Panel(
                str(error.response.json()),
                title='Error Details',
                border_style='red'
            ))
        else:
            self.console.print(f'\n[red]Error: {str(error)}[/red]')

    def get_user_input(self) -> str:
        return Prompt.ask('\n[bold green]You[/bold green]')
