import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import tiktoken

from src.domain.document import DocMetadata, Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkMetrics:
    total_chunks: int
    avg_size: float
    median_size: float
    min_size: int
    max_size: int


class TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int:
        ...


class TiktokenCounter:
    def __init__(self, model_name: str = 'cl100k_base'):
        self.tokenizer = tiktoken.get_encoding(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


class TextFormatter:
    @staticmethod
    def format_for_tokenization(text: str) -> str:
        return f'<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant<|im_end|>'


class HeaderExtractor:
    @staticmethod
    def extract_headers(text: str) -> Dict[str, List[str]]:
        headers: Dict[str, List[str]] = {}
        header_regex = r'(^|\n)(#{1,6})\s+(.*)'

        for match in re.finditer(header_regex, text, re.MULTILINE):
            level = len(match.group(2))
            content = match.group(3).strip()
            key = f'h{level}'
            if key not in headers:
                headers[key] = []
            headers[key].append(content)

        return headers

    @staticmethod
    def update_headers(current: Dict[str, List[str]], new_headers: Dict[str, List[str]]) -> None:
        for level in range(1, 7):
            key = f'h{level}'
            if key in new_headers:
                current[key] = new_headers[key]
                HeaderExtractor._clear_lower_headers(current, level)

    @staticmethod
    def _clear_lower_headers(headers: Dict[str, List[str]], level: int) -> None:
        for header in range(level + 1, 7):
            headers.pop(f'h{header}', None)


class URLProcessor:
    def __init__(self) -> None:
        self.url_pattern = r'https?://[\w\-\.]+\.[a-zA-Z]{2,}[^\s\)]*'
        self.urls: List[str] = []
        self.images: List[str] = []
        self.url_index = 0
        self.image_index = 0

    def process_content(self, text: str) -> Tuple[str, List[str], List[str]]:
        content = text
        content = self._process_images(content)
        content = self._process_markdown_urls(content)
        content = self._process_standalone_urls(content)

        logger.debug(f'Extracted {len(self.urls)} URLs and {len(self.images)} images')
        return content, list(dict.fromkeys(self.urls)), list(dict.fromkeys(self.images))

    def _process_images(self, content: str) -> str:
        return re.sub(
            r'!\s*(?:\[[^\]]*\])?\s*\(([^)]+)\)',
            self._replace_image,
            content
        )

    def _process_markdown_urls(self, content: str) -> str:
        return re.sub(
            r'\[[^\]]*\]\(([^)]+)\)',
            self._replace_url,
            content
        )

    def _process_standalone_urls(self, content: str) -> str:
        return re.sub(self.url_pattern, self._replace_standalone_url, content)

    def _replace_image(self, match: re.Match) -> str:
        img_url = re.search(self.url_pattern, match.group(0))
        if img_url:
            url = img_url.group(0)
            self.images.append(url)
            result = f'![]({{$img{self.image_index}}})'
            self.image_index += 1
            return result
        return match.group(0)

    def _replace_url(self, match: re.Match) -> str:
        url_match = re.search(self.url_pattern, match.group(0))
        if url_match:
            url = url_match.group(0)
            if url not in self.images:
                self.urls.append(url)
                result = f'{{$url{self.url_index}}}'
                self.url_index += 1
                return result
        return match.group(0)

    def _replace_standalone_url(self, match: re.Match) -> str:
        url = match.group(0)
        if url not in self.images and url not in self.urls:
            self.urls.append(url)
            result = f'{{$url{self.url_index}}}'
            self.url_index += 1
            return result
        return url


class ChunkStrategy(ABC):
    @abstractmethod
    def get_chunk(self, text: str, start: int, limit: int, token_counter: TokenCounter) -> Tuple[str, int]:
        pass


class NewlineChunkStrategy(ChunkStrategy):
    def get_chunk(self, text: str, start: int, limit: int, token_counter: TokenCounter) -> Tuple[str, int]:
        formatter = TextFormatter()
        remaining_text = text[start:]

        if not remaining_text:
            return '', start

        end = start + min(len(remaining_text), limit * 4)
        chunk_text = text[start:end]

        max_iterations = 100
        iteration = 0

        current_tokens = token_counter.count_tokens(formatter.format_for_tokenization(chunk_text))

        while current_tokens != limit and iteration < max_iterations:
            if current_tokens > limit:
                end = start + (end - start) // 2
            else:
                potential_end = min(len(text), end + (end - start) // 2)
                if potential_end == end:
                    break
                end = potential_end

            chunk_text = text[start:end]
            current_tokens = token_counter.count_tokens(formatter.format_for_tokenization(chunk_text))
            iteration += 1

            if end <= start + 1:
                break

        if iteration >= max_iterations:
            logger.warning(f'Hit maximum iterations ({max_iterations}) while trying to find chunk size.')
            return text[start:start + max(1, limit)], start + max(1, limit)

        end = self._adjust_to_newline(text, start, end, chunk_text, token_counter, limit)
        return text[start:end], end

    def _adjust_to_newline(
        self,
        text: str,
        start: int,
        end: int,
        chunk_text: str,
        token_counter: TokenCounter,
        limit: int,
    ) -> int:
        formatter = TextFormatter()

        next_nl = text.find('\n', end)
        if next_nl != -1 and next_nl < len(text):
            extended_text = text[start:next_nl + 1]
            if token_counter.count_tokens(
                    formatter.format_for_tokenization(extended_text)) <= limit * 1.2:
                return next_nl + 1

        prev_nl = text.rfind('\n', start, end)
        if prev_nl > start:
            reduced_text = text[start:prev_nl + 1]
            if token_counter.count_tokens(formatter.format_for_tokenization(reduced_text)) > 0:
                return prev_nl + 1

        return end


class TextSplitter:
    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        chunk_strategy: Optional[ChunkStrategy] = None,
    ):
        self.token_counter = token_counter or TiktokenCounter()
        self.chunk_strategy = chunk_strategy or NewlineChunkStrategy()
        self.header_extractor = HeaderExtractor()
        self.url_processor = URLProcessor()

    def split(self, text: str, limit: int) -> List[Document]:
        logger.info(f'Starting split process with limit: {limit} tokens')
        chunks: List[Document] = []
        position = 0
        current_headers: Dict[str, List[str]] = {}

        while position < len(text):
            logger.debug(f'Processing chunk starting at position: {position}')

            chunk_text, chunk_end = self.chunk_strategy.get_chunk(text, position, limit, self.token_counter)
            if not chunk_text:
                break

            headers = self.header_extractor.extract_headers(chunk_text)
            self.header_extractor.update_headers(current_headers, headers)
            content, urls, images = self.url_processor.process_content(chunk_text)

            chunks.append(Document(
                text=content,
                metadata=DocMetadata(
                    tokens=self.token_counter.count_tokens(chunk_text),
                    headers=dict(current_headers),
                    urls=urls,
                    images=images
                ),
                start=position,
                end=chunk_end,
            ))

            position = chunk_end

        logger.info(f'Split process completed. Total chunks: {len(chunks)}')
        return chunks


class FileProcessor:
    def __init__(self, splitter: TextSplitter):
        self.splitter = splitter

    def process_file(self, file_path: str, token_limit: int = 1000) -> dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        docs = self.splitter.split(text, token_limit)

        json_path = str(Path(file_path).with_suffix('.json'))
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump([doc.to_dict() for doc in docs], file, indent=2)

        metrics = self._calculate_metrics([doc.metadata.tokens for doc in docs])

        return {
            'file': os.path.basename(file_path),
            'avgChunkSize': f'{metrics.avg_size:.2f}',
            'medianChunkSize': metrics.median_size,
            'minChunkSize': metrics.min_size,
            'maxChunkSize': metrics.max_size,
            'totalChunks': metrics.total_chunks
        }

    def _calculate_metrics(self, chunk_sizes: List[int]) -> ChunkMetrics:
        if not chunk_sizes:
            return ChunkMetrics(0, 0.0, 0.0, 0, 0)

        return ChunkMetrics(
            total_chunks=len(chunk_sizes),
            avg_size=sum(chunk_sizes) / len(chunk_sizes),
            median_size=sorted(chunk_sizes)[len(chunk_sizes) // 2],
            min_size=min(chunk_sizes),
            max_size=max(chunk_sizes)
        )
