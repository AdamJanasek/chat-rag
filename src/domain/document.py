from typing import Dict, List

from pydantic import BaseModel


class DocMetadata(BaseModel):
    tokens: int
    headers: Dict[str, List[str]]
    urls: List[str]
    images: List[str]


class Document(BaseModel):
    text: str
    metadata: DocMetadata
    start: int
    end: int

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'metadata': {
                'tokens': self.metadata.tokens,
                'headers': self.metadata.headers,
                'urls': self.metadata.urls,
                'images': self.metadata.images
            }
        }
