from typing import Any, Dict

from src.services.base.ai_service import AIService
from src.services.gpt import OpenAIService
from src.services.ollama import OllamaService


def create_ai_service(provider: str = 'openai') -> AIService:
    if provider == 'openai':
        return OpenAIService()
    elif provider == 'ollama':
        return OllamaService()
    else:
        raise ValueError(f'Unknown AI service provider: {provider}')


def format_search_result(result: Dict[str, Any]) -> str:
    filename = result['payload'].get('filename', 'unknown')
    text = result['payload']['text']
    return f'[{filename}]: {text}'
