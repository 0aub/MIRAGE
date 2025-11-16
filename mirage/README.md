# MIRAGE Backend

**M**ultilingual **I**nformation **R**etrieval with **A**ccelerated **G**raph **E**mbeddings

## Overview

MIRAGE is an advanced RAG system that combines:
- **Graph-RAG**: Dynamic knowledge graph construction from documents
- **REFRAG Technology**: 30.85× faster inference through intelligent compression
- **Arabic Language Support**: Optimized for Arabic with multilingual capabilities
- **LangGraph Orchestration**: Complex pipeline management
- **Claude API Integration**: High-quality response generation

## Architecture

```
mirage/
├── src/
│   ├── api/                      # FastAPI services
│   │   ├── document_service.py   # Document upload & processing
│   │   ├── chat_service.py       # Chat with Graph-RAG
│   │   ├── graph_service.py      # Graph exploration
│   │   └── refrag_service.py     # Compression metrics
│   │
│   ├── core/                     # Business logic
│   │   ├── document_processor/   # PDF, HTML, JSON processing
│   │   ├── graph_builder/        # Entity extraction, Neo4j
│   │   ├── refrag/               # REFRAG compression
│   │   ├── embeddings/           # Jina v3 wrapper
│   │   └── langgraph_pipeline/   # Workflow orchestration
│   │
│   ├── models/                   # Pydantic models
│   ├── config/                   # Configuration
│   └── utils/                    # Helpers
│
├── tests/                        # Test suite
├── Dockerfile                    # Docker configuration
├── requirements.txt              # Python dependencies
└── main.py                       # Application entry point
```

## API Services

### 1. Document Service (`/document`)
- Upload documents (PDF, HTML, JSON)
- Track processing status
- Manage document lifecycle

### 2. Chat Service (`/chat`)
- Send messages and get AI responses
- WebSocket support for streaming
- Conversation history management
- **Pipeline**: Query → Graph Retrieval → REFRAG → Claude API

### 3. Graph Service (`/graph`)
- Explore knowledge graph
- Search for entities
- View graph statistics
- Find paths between nodes

### 4. REFRAG Service (`/refrag`)
- Compression metrics
- RL policy statistics
- Cache management
- Configuration control

## Quick Start

### 1. Set up environment
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Run with Docker
```bash
# From the root directory
docker-compose up mirage
```

### 3. Access API
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## Development Phases

### Phase 1: Document Processing ✅ Structure Ready
- PDF text extraction (PyMuPDF)
- HTML content extraction (BeautifulSoup)
- JSON intelligent field extraction
- Text chunking and normalization

### Phase 2: Graph Construction 🚧 To Implement
- Arabic NER (CAMeLTools)
- English NER (spaCy)
- Relationship extraction
- Neo4j integration

### Phase 3: REFRAG Integration 🚧 To Implement
- Jina v3 embedding wrapper
- REFRAG compression implementation
- RL policy training
- Cache optimization

### Phase 4: LangGraph Pipeline 🚧 To Implement
- Workflow state management
- Graph retrieval nodes
- Compression nodes
- Claude API integration

### Phase 5: API Services ✅ Structure Ready
- FastAPI endpoints
- WebSocket chat
- Background task processing
- Monitoring and metrics

## Key Technologies

- **FastAPI**: High-performance async API framework
- **LangChain/LangGraph**: Workflow orchestration
- **Neo4j**: Graph database for knowledge relationships
- **Qdrant**: Vector database for embeddings
- **Redis**: Caching and task queues
- **Jina v3**: Multilingual embeddings
- **Claude API**: Response generation
- **CAMeLTools**: Arabic NLP
- **spaCy**: English NLP

## Configuration

Key environment variables (see `.env.example`):

```bash
# API Keys
CLAUDE_API_KEY=your_key
JINA_API_KEY=your_key

# Databases
NEO4J_URI=bolt://neo4j:7687
QDRANT_HOST=qdrant
REDIS_URL=redis://redis:6379

# REFRAG
REFRAG_COMPRESSION_RATE=16
```

## Next Steps

1. Implement document processors (Phase 1)
2. Build entity extraction pipeline (Phase 2)
3. Integrate REFRAG from GitHub (Phase 3)
4. Create LangGraph workflow (Phase 4)
5. Connect all services (Phase 5)

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src tests/

# Lint code
black src/
ruff src/
```

## Monitoring

- Prometheus metrics: `/metrics`
- Health check: `/health`
- API documentation: `/docs`

## License

[Your License]

## Contributing

[Contributing guidelines]
