# RAG Blueprint with Vespa

A complete Retrieval-Augmented Generation (RAG) application built with [Vespa](https://vespa.ai/) and FastAPI, using **pyvespa** for declarative configuration.

## 🚀 Features

- **Complete Vespa RAG Stack** - OpenAI LLM integration, Nomic ModernBERT embeddings (768-dim), hybrid search
- **Python-First Configuration** - Services, schemas, and profiles defined using pyvespa (no XML!)
- **Automatic Deployment** - One command to deploy Vespa and feed 100 sample documents
- **Multiple Query Modes** - Hybrid search, RAG generation, and deep research
- **Production-Ready** - FastAPI with health checks, batch operations, and comprehensive error handling

## 📋 Prerequisites

- Docker installed and running
- Python 3.10+
- `uv` or `pip` for dependency management

## ⚡ Quick Start

```bash
# Install dependencies
uv sync

# Start the application (deploys Vespa + feeds documents automatically)
python main.py
```

That's it! The application will:
1. Deploy Vespa to Docker
2. Configure services, schemas, and profiles using pyvespa
3. Automatically feed 100 sample documents
4. Start FastAPI server at `http://localhost:8000`

## 🔍 Try a Query

Once started, query your documents:

```bash
# Hybrid search (BM25 + vector similarity)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "kubernetes deployment best practices",
    "query_profile": "hybrid",
    "hits": 10
  }'
```

Or visit the interactive API docs at `http://localhost:8000/docs`

## 🏗️ Architecture

### Services Configuration (Python, not XML!)

The entire Vespa services configuration is defined in Python using pyvespa:

**`app/services/services.py`** - Python equivalent of `services.xml`:
- OpenAI LLM client component
- Nomic ModernBERT embedder (768 dimensions)
- RAG search chain
- Document processing pipeline
- Content cluster configuration

```python
from vespa.configuration.services import (
    services, container, component, search, chain,
    searcher, content, transformer_model, ...
)

# Define services in Python instead of XML
services_config = services(
    container(...),
    content(...),
    version="1.0",
    minimum_required_vespa_version="8.519.55"
)
```

### Document Schema

**`app/schemas/doc.py`** - Rich document schema with:
- Full-text search fields (title, chunks)
- Vector embeddings (title_embedding, chunk_embeddings)
- HNSW indexes for fast similarity search
- Metadata fields (timestamps, favorite, open_count)

### Rank Profiles

**`app/profiles/`** - Multiple ranking strategies:
- `base-features` - Feature extraction
- `learned-linear` - Linear model with learned weights
- `collect-training-data` - For model training
- `collect-second-phase` - Second-phase ranking

### Query Profiles

**`app/search/query_profiles/`** - Pre-configured search modes:
- `hybrid` - BM25 text + vector similarity (default)
- `rag` - Retrieval-augmented generation with OpenAI
- `deepresearch` - Enhanced retrieval for research tasks

## 📁 Project Structure

```
rag-blueprint-vespa/
├── main.py                          # FastAPI app with auto-deploy & auto-feed
├── app/
│   ├── services/
│   │   ├── services.py             # Vespa services (pyvespa, replaces services.xml)
│   │   └── services.xml            # Original XML reference
│   ├── schemas/
│   │   └── doc.py                  # Document schema definition
│   ├── profiles/                   # Rank profile definitions
│   ├── search/
│   │   └── query_profiles/         # Query profile configurations
│   └── dataset/
│       └── docs.jsonl              # 100 sample documents
├── USAGE.md                        # Detailed usage guide
└── README.md                       # This file
```

## 🎯 Key Endpoints

- `GET /` - Health check and status
- `GET /health` - Detailed health information
- `POST /query` - Search documents with different profiles
- `POST /feed` - Feed a single document
- `POST /batch-feed` - Re-feed all documents from dataset
- `GET /vespa/status` - Vespa application status

## 📊 Dataset

Includes 100 diverse documents:
- Technical documentation (Python implementations, architecture docs)
- Meeting notes and investor updates
- Research notes and training logs
- Business planning documents

Each document has rich metadata (timestamps, usage stats, favorites).

## 🔧 Configuration Highlights

### Services (pyvespa-based)

The `app/services/services.py` file demonstrates the **services.xml to Python conversion**:

**Before (XML):**
```xml
<component id="nomicmb" type="hugging-face-embedder">
    <transformer-model url="..." />
    <tokenizer-model url="..." />
    <transformer-output>token_embeddings</transformer-output>
    <max-tokens>8192</max-tokens>
</component>
```

**After (Python with pyvespa):**
```python
component(
    transformer_model(url="..."),
    tokenizer_model(url="..."),
    transformer_output("token_embeddings"),
    max_tokens("8192"),
    id="nomicmb",
    type_="hugging-face-embedder"
)
```

All configuration is now Python code - type-safe, composable, and version-controlled!

## 🎓 Learn More

- **Detailed Usage Guide**: See [USAGE.md](USAGE.md)
- **Vespa Documentation**: https://docs.vespa.ai/
- **pyvespa Documentation**: https://pyvespa.readthedocs.io/

## 📝 License

This project demonstrates Vespa RAG capabilities. Refer to individual component licenses:
- Vespa: Apache 2.0
- FastAPI: MIT

## 🤝 Contributing

This is a blueprint/template project. Feel free to fork and customize for your use case!

## 🙏 Acknowledgments

Built with:
- [Vespa.ai](https://vespa.ai/) - The open-source big data serving engine
- [pyvespa](https://pyvespa.readthedocs.io/) - Python API for Vespa
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- OpenAI & Nomic AI - LLM and embedding models
