# MIRAGE V2

**M**ultilingual **I**nformation **R**etrieval with **A**ccelerated **G**raph **E**mbeddings

A sophisticated GraphRAG system combining vector search with knowledge graph reasoning for enhanced Arabic/English information retrieval.

## Status

**Current Grade: A-** (4/5 retrieval tests passed, 100% relevance rate)

| Component | Status |
|-----------|--------|
| Vector Search (Qdrant) | ✅ Operational |
| Knowledge Graph (Neo4j) | ✅ 810 entities, 3,223 relationships |
| Query Routing | ✅ 5 modes (naive/local/global/hybrid/mix) |
| LLM Generation (TGI) | ✅ Qwen3-4B local inference |
| Arabic Support | ✅ Full Arabic NLP |

## Architecture

```
                    Query
                      │
              ┌───────▼───────┐
              │ Query Router  │ (Arabic/English pattern matching)
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
    ┌───────┐    ┌────────┐    ┌────────┐
    │ Naive │    │ Local  │    │ Global │
    │Vector │    │Entity  │    │Relation│
    │Search │    │Search  │    │Search  │
    └───┬───┘    └───┬────┘    └───┬────┘
        └─────────────┼─────────────┘
                      ↓
              ┌───────────────┐
              │ RRF Fusion    │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ TGI (Qwen3)   │
              └───────┬───────┘
                      ↓
                   Answer
```

## Quick Start

```bash
# Start all services
docker compose up -d

# Access interfaces
open http://localhost:3000      # Web UI
open http://localhost:8000/docs # API Docs
open http://localhost:7474      # Neo4j Browser
```

## API Endpoints

### Chat/Ask (V2)
```bash
curl -X POST http://localhost:8000/chat/ask \
  -H "Content-Type: application/json" \
  -d '{
    "message": "ما هي جائزة الحكومة الرقمية؟",
    "retrieval_mode": "auto",
    "top_k": 5
  }'
```

### Retrieval Modes
- `auto` - Automatic mode selection based on query
- `naive` - Simple vector similarity
- `local` - Entity-focused (query → entities → chunks)
- `global` - Relationship-focused (query → relationships → entities → chunks)
- `hybrid` - Combines local + global
- `mix` - All modes with RRF fusion

### Ingest Content
```bash
# YouTube Video
curl -X POST http://localhost:8000/url/process-async \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'

# Web URL
curl -X POST http://localhost:8000/url/process-async \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

## Project Structure

```
MIRAGE/
├── mirage/                    # Backend Python application
│   └── src/
│       ├── api/              # FastAPI routes
│       ├── core/
│       │   ├── retrieval/    # V2 retrieval engine
│       │   ├── generation/   # Prompt management
│       │   ├── graph_builder/# Neo4j integration
│       │   └── vector_store/ # Qdrant integration
│       └── config/           # Settings
├── ui/                       # React frontend
├── docs/                     # Documentation
│   ├── graphrag/            # GraphRAG research & plans
│   └── archives/            # Historical docs
├── docker-compose.yml
└── EVALUATION_REPORT.md      # Latest evaluation results
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.11 |
| Vector DB | Qdrant |
| Graph DB | Neo4j |
| LLM | TGI (Qwen3-4B) |
| Embeddings | paraphrase-multilingual-mpnet |
| Frontend | React, TypeScript, Vite |
| Cache | Redis |

## Documentation

- [Docker Setup](docs/DOCKER_SETUP.md)
- [TGI Setup](docs/TGI_SETUP.md)
- [UI Integration](docs/UI_INTEGRATION_GUIDE.md)
- [GraphRAG Research](docs/graphrag/)
- [Evaluation Report](EVALUATION_REPORT.md)

## Development

```bash
# Backend
cd mirage && pip install -r requirements.txt
uvicorn src.api.main:app --reload

# Frontend
cd ui && npm install && npm run dev
```

## License

MIT
