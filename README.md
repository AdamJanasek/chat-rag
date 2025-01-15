# RAG API for AI Course

A FastAPI-based Retrieval-Augmented Generation (RAG) system that processes MD documents, stores them in a vector database, and enables semantic search with reranking capabilities.

## Features

- Pre-loaded AI course content in vector database
- Document processing and chunking with intelligent text splitting
- Vector embeddings storage using Qdrant
- Semantic search with optional reranking
- Interactive console interface for chat-like interactions
- Docker containerization for easy deployment
- URL and image reference extraction from documents
- Token-aware text chunking with configurable strategies
- Header hierarchy preservation in document processing

## System Architecture

The system consists of several key components:

1. **FastAPI Backend**
   - REST API endpoints for document upload and querying
   - Dependency injection container for service management

2. **Vector Service**
   - Manages interactions with Qdrant vector database
   - Handles embedding creation and storage
   - Implements semantic search with optional reranking

3. **Text Processing**
   - Smart text chunking with token awareness
   - URL and image extraction
   - Header hierarchy maintenance
   - Configurable chunking strategies

4. **Console Interface**
   - Interactive chat-like experience
   - HTTP-based communication with the API
   - Rich console output
   - Message history management

## Docker Services

The application is containerized using Docker Compose with the following services:

```yaml
services:
  app:
    build:
      context: ./
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  console:
    build:
      context: ./
      dockerfile: docker/Dockerfile.console
    volumes:
      - .:/app
    depends_on:
      - app
    stdin_open: true
    tty: true

volumes:
  qdrant_data:
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Environment variables configured (see Configuration section)

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   ```

2. Create and configure the `.env` file:
   ```env
   # Add your environment variables here
   ```

3. Run the application:
   ```bash
   ./run.sh
   ```

4. After first run, import the course data snapshot:
   ```bash
   # Copy snapshot to Qdrant container
   docker cp qdrant_snapshot/ai_course_docs.snapshot rag-app-qdrant-1:/qdrant/storage/

   # Execute snapshot import in Qdrant
   docker exec rag-app-qdrant-1 qdrant snapshot restore --snapshot ai_course_docs.snapshot
   ```

This will:

- Start all required services using Docker Compose
- Build the console application
- Launch the interactive chat interface with access to pre-loaded AI course content

The application comes with pre-loaded AI course materials in the vector database (located in `storage/qdrant`), allowing you to immediately start querying and interacting with the course content. You can also upload additional documents to enhance the knowledge base.

## API Endpoints

### POST /upload
Upload and process a document for vector storage.

**Request:**
- Multipart form data with file

**Response:**
```json
{
    "status": "success",
    "message": "Document processed successfully",
    "metrics": {
        "avgChunkSize": "850.25",
        "totalChunks": 10
        // ... other metrics
    }
}
```

### POST /query
Query the vector database for relevant document chunks.

**Request:**
```json
{
    "query": "your search query",
    "top_k": 3,
    "rerank": true
}
```

**Response:**
```json
{
    "results": [
        {
            "text": "matched text",
            "score": 0.95,
            "metadata": {
                // document metadata
            }
        }
    ]
}
```

## Configuration

The application can be configured using environment variables or command-line arguments for the console interface:

- Console Options:
  - `--top-k`: Number of top results to consider (default: 3)
  - `--rerank/--no-rerank`: Enable/disable reranking
  - `--base-url`: API base URL (default: http://app:8000)

## Development

### Project Structure

```
src/
├── api/            # FastAPI application code
├── services/       # Core business logic
├── domain/         # Domain models and types
├── console/        # Interactive console interface
└── settings.py     # Application settings
```

## License

MIT License

Copyright (c) 2025 Adam Janasek
