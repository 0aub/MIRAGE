# MIRAGE

**M**ultilingual **I**nformation **R**etrieval with **A**ccelerated **G**raph **E**mbeddings

A sophisticated knowledge graph and retrieval system combining vector search with graph-based reasoning for enhanced information retrieval and analysis.

![MIRAGE Logo](logo.png)

## Overview

MIRAGE is an advanced Retrieval-Augmented Generation (RAG) system that implements GraphRAG principles by combining:
- **Vector Database** (Qdrant) for semantic similarity search
- **Knowledge Graph** (Neo4j) for relationship-based reasoning
- **Multi-modal Processing** for various content types (text, PDF, YouTube, URLs)
- **Intelligent Chunking** with content-type-specific strategies
- **LLM-powered Analysis** for entity extraction and relationship mapping

## Features

### Core Capabilities
- **Multi-source Ingestion**: Process PDFs, web URLs, YouTube videos, and text documents
- **Hybrid Search**: Combine vector similarity with graph-based reasoning
- **Entity Extraction**: Automatic identification of entities and relationships
- **Content-Aware Chunking**: Specialized strategies for different content types
- **Interactive Chat**: Query your knowledge base with conversational AI
- **Graph Visualization**: Explore knowledge graphs with Neo4j Browser

### Advanced Processing
- Semantic chunking for coherent context windows
- Entity and relationship extraction using LLMs
- Multi-hop reasoning through graph traversal
- Configurable chunking strategies per content type
- Support for multiple LLM providers (OpenAI, Anthropic, TGI)

## Architecture

```
┌─────────────────┐
│   Web UI        │  React + TypeScript + shadcn/ui
│   (Port 3000)   │
└────────┬────────┘
         │
┌────────▼────────┐
│   FastAPI       │  Python REST API
│   (Port 8000)   │
└────────┬────────┘
         │
    ┌────┴────────────────────┐
    │                         │
┌───▼─────┐            ┌──────▼──────┐
│ Qdrant  │            │   Neo4j     │
│ Vectors │            │   Graph     │
│ (6333)  │            │   (7474)    │
└─────────┘            └─────────────┘
```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- At least one LLM API key (OpenAI, Anthropic, or TGI endpoint)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MIRAGE
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Start services**
   ```bash
   docker compose up -d
   ```

4. **Access the interfaces**
   - Web UI: http://localhost:3000
   - API Docs: http://localhost:8000/docs
   - Neo4j Browser: http://localhost:7474
   - Qdrant Dashboard: http://localhost:6333/dashboard

## Project Structure

```
MIRAGE/
├── mirage/              # Backend Python application
│   ├── src/
│   │   ├── api/         # FastAPI routes and services
│   │   ├── core/        # Core processing logic
│   │   ├── models/      # Data models
│   │   ├── config/      # Configuration and prompts
│   │   └── utils/       # Utility functions
│   └── tests/           # Backend tests
├── ui/                  # Frontend React application
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   └── lib/         # Utilities
│   └── public/          # Static assets
├── docs/                # Documentation
│   ├── archives/        # Historical documentation
│   ├── ARCHITECTURE_REDESIGN.md
│   ├── DOCKER_SETUP.md
│   ├── GRAPHRAG_ENHANCEMENTS.md
│   └── ...
├── scripts/             # Utility scripts
├── docker-compose.yml   # Docker orchestration
└── README.md           # This file
```

## Configuration

### LLM Providers
Configure in `mirage/src/config/settings.yaml`:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Text Generation Inference (TGI) for local models

### Chunking Strategies
Customize chunking per content type:
- **Text**: Semantic chunking with configurable overlap
- **PDF**: Structure-aware chunking
- **YouTube**: Timestamp-based semantic chunking
- **URL**: Content-aware chunking

### Processing Pipeline
Configure in settings:
- Entity extraction prompts
- Relationship extraction rules
- Content rewriting options
- Token constraints

## Usage

### Adding Documents

**Via UI:**
1. Navigate to http://localhost:3000
2. Select content type (Text, PDF, URL, YouTube)
3. Enter content or upload file
4. Process and wait for completion

**Via API:**
```bash
curl -X POST http://localhost:8000/db/process \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your text here",
    "content_type": "text"
  }'
```

### Querying

**Chat Interface:**
```bash
curl -X POST http://localhost:8000/db/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is...?"}]
  }'
```

### Viewing the Graph

Visit http://localhost:7474 and run Cypher queries:
```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
```

## Development

### Backend Development
```bash
cd mirage
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

### Frontend Development
```bash
cd ui
npm install
npm run dev
```

### Running Tests
```bash
cd mirage
pytest tests/
```

## Documentation

- [Architecture Overview](docs/ARCHITECTURE_REDESIGN.md)
- [Docker Setup Guide](docs/DOCKER_SETUP.md)
- [GraphRAG Enhancements](docs/GRAPHRAG_ENHANCEMENTS.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [TGI Setup](docs/TGI_SETUP.md)
- [UI Integration](docs/UI_INTEGRATION_GUIDE.md)

## Technology Stack

**Backend:**
- FastAPI
- Python 3.11+
- Neo4j (Graph Database)
- Qdrant (Vector Database)
- Redis (Caching)

**Frontend:**
- React 18
- TypeScript
- Vite
- shadcn/ui
- TailwindCSS

**AI/ML:**
- OpenAI API
- Anthropic API
- Text Generation Inference
- Sentence Transformers

## Contributing

Contributions are welcome! Please read the contribution guidelines before submitting PRs.

## License

[Specify your license here]

## Acknowledgments

Built with inspiration from Microsoft's GraphRAG and modern RAG architectures.
