from src.domain.document import Document
from src.splitters.text_splitter import (
    HeaderExtractor,
    NewlineChunkStrategy,
    TextSplitter,
    TiktokenCounter,
    URLProcessor,
)

SAMPLE_TEXT = '''# Header 1
This is a paragraph with a [link](https://example.com).
## Header 2
Another paragraph with an ![image](https://example.com/image.jpg).
### Header 3
Final paragraph with https://standalone-url.com.'''

SAMPLE_MARKDOWN = '''# Main Title
## Section 1
Content for section 1
## Section 2
Content for section 2
### Subsection 2.1
More content here'''


def mock_count_tokens(text: str) -> int:
    return len(text.split())


def get_test_splitter() -> TextSplitter:
    counter = TiktokenCounter('cl100k_base')
    counter.count_tokens = mock_count_tokens  # type: ignore
    return TextSplitter(token_counter=counter)


def test_tiktoken_counter():
    counter = TiktokenCounter()
    tokens = counter.count_tokens('Hello world')
    assert tokens > 0


def test_text_splitter_empty_input():
    splitter = get_test_splitter()
    result = splitter.split('', 1000)
    assert len(result) == 0


def test_text_splitter_simple_text():
    splitter = get_test_splitter()
    text = 'This is a simple test text'
    chunks = splitter.split(text, 5)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)


def test_header_extraction():
    text = '# Header 1\nContent\n## Header 2'
    headers = HeaderExtractor.extract_headers(text)
    assert 'h1' in headers
    assert headers['h1'] == ['Header 1']
    assert 'h2' in headers
    assert headers['h2'] == ['Header 2']


def test_header_update():
    current = {'h1': ['Old'], 'h2': ['Old Sub']}
    new = {'h1': ['New']}
    HeaderExtractor.update_headers(current, new)
    assert current['h1'] == ['New']
    assert 'h2' not in current


def test_url_processing():
    processor = URLProcessor()
    text = 'Check [this](https://example.com) and ![img](https://example.com/img.jpg)'
    content, urls, images = processor.process_content(text)
    assert 'https://example.com' in urls
    assert 'https://example.com/img.jpg' in images
    assert '{$url0}' in content
    assert '{$img0}' in content


def test_newline_chunk_strategy():
    strategy = NewlineChunkStrategy()
    counter = TiktokenCounter()
    counter.count_tokens = mock_count_tokens  # type: ignore
    text = 'Line 1\nLine 2\nLine 3'
    chunk, end = strategy.get_chunk(text, 0, 10, counter)
    assert 'Line 1' in chunk
    assert end > 0


def test_full_document_splitting():
    splitter = get_test_splitter()
    chunks = splitter.split(SAMPLE_MARKDOWN, 10)

    assert len(chunks) > 0
    first_chunk = chunks[0]

    assert isinstance(first_chunk, Document)
    assert first_chunk.metadata.headers
    assert isinstance(first_chunk.metadata.tokens, int)
    assert first_chunk.start >= 0
    assert first_chunk.end > first_chunk.start


def test_headers_maintained_across_chunks():
    splitter = get_test_splitter()
    chunks = splitter.split(SAMPLE_MARKDOWN, 10)

    for chunk in chunks:
        assert chunk.metadata.headers is not None
        if '# Main Title' in chunk.text:
            assert 'h1' in chunk.metadata.headers


def test_chunk_size_limits():
    splitter = get_test_splitter()
    limit = 5
    chunks = splitter.split(SAMPLE_TEXT, limit)

    for chunk in chunks:
        tokens = chunk.metadata.tokens
        assert tokens <= limit, f'Chunk exceeds token limit: {tokens} > {limit}'
