import os
from typing import Optional

from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    AI_PROVIDER: str = os.getenv('AI_PROVIDER', 'openai')
    QDRANT_HOST: str = os.getenv('QDRANT_HOST', 'qdrant')
    QDRANT_PORT: int = int(os.getenv('QDRANT_PORT', '6333'))
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')

    class Config:
        env_file = '../.env'
