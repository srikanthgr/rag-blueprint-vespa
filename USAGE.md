# RAG Blueprint with Vespa - Usage Guide

This guide explains how to deploy the Vespa RAG application and use it for document search.

## Prerequisites

- Docker installed and running
- Python 3.10+
- Dependencies installed: `uv sync` or `pip install -r requirements.txt`

## Quick Start

Start the FastAPI application which automatically deploys Vespa and feeds documents:

```bash
python main.py
```

The application will automatically:
1. Build the complete Vespa application package
2. Deploy it to Docker (takes 2-5 minutes on first run)
3. **Automatically feed all 100 documents from `app/dataset/docs.jsonl`**
4. Start FastAPI server on `http://localhost:8000`

Once running, you can:
- Visit the API docs at `http://localhost:8000/docs`
- Check health at `http://localhost:8000/health`
- Query documents immediately!

### Startup Log Example

```
INFO - Starting up: Building Vespa application package...
INFO - Deploying Vespa application to Docker...
INFO - Successfully deployed Vespa application at http://localhost:8080
INFO - Auto-feeding documents from dataset...
INFO - Successfully read 100 documents from app/dataset/docs.jsonl
INFO - Starting to feed 100 documents to Vespa...
INFO - Progress: 10/100 documents processed
INFO - Progress: 20/100 documents processed
...
INFO - Batch feed completed: 100 successful, 0 failed
INFO - Auto-feed result: 100/100 documents loaded
```

## Re-feeding Documents

If you need to manually re-feed documents (e.g., after adding new ones to the dataset):

```bash
curl -X POST http://localhost:8000/batch-feed
```

Or use the interactive API docs at `http://localhost:8000/docs` and click on the `/batch-feed` endpoint.

## Querying Documents

### Using the API

Query documents using different query profiles:

**Hybrid Search (BM25 + Vector Similarity):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "kubernetes deployment",
    "query_profile": "hybrid",
    "hits": 10
  }'
```

**RAG Search (with OpenAI LLM):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best practices for scaling B2B sales?",
    "query_profile": "rag",
    "hits": 50
  }'
```

**Deep Research:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "RAG architecture implementation",
    "query_profile": "deepresearch",
    "hits": 10
  }'
```

### Query Profiles

The application supports three query profiles:

1. **`hybrid`** - Combines BM25 text search with vector similarity search
   - Best for: General search with semantic understanding
   - Returns: Top matched documents with relevance scores

2. **`rag`** - Retrieval-Augmented Generation with OpenAI
   - Best for: Natural language questions requiring synthesized answers
   - Returns: Generated response based on retrieved context (requires OpenAI API key)

3. **`deepresearch`** - Enhanced retrieval for research tasks
   - Best for: Deep dives into specific topics
   - Returns: Comprehensive document matches

## API Endpoints

### Health & Status

- `GET /` - Root endpoint with basic status
- `GET /health` - Detailed health check
- `GET /vespa/status` - Vespa application status and available endpoints

### Document Operations

- `POST /feed` - Feed a single document
  ```json
  {
    "id": "doc-123",
    "fields": {
      "id": "doc-123",
      "title": "Document Title",
      "text": "Document content...",
      "created_timestamp": 1234567890,
      "modified_timestamp": 1234567890,
      "last_opened_timestamp": 1234567890,
      "open_count": 0,
      "favorite": false
    }
  }
  ```

- `POST /batch-feed` - Feed all documents from `app/dataset/docs.jsonl`

- `POST /query` - Query documents
  ```json
  {
    "query": "search query text",
    "query_profile": "hybrid",
    "hits": 10
  }
  ```

## Dataset

The dataset (`app/dataset/docs.jsonl`) contains 100 documents including:
- Technical documentation (code implementations, architecture docs)
- Meeting notes and research notes
- Training logs and personal notes
- Business planning documents

Each document has:
- `id`: Unique identifier
- `title`: Document title
- `text`: Main content (markdown formatted)
- `created_timestamp`: Creation time (Unix timestamp)
- `modified_timestamp`: Last modification time
- `last_opened_timestamp`: Last access time
- `open_count`: Number of times opened
- `favorite`: Boolean flag

## Architecture Components

The deployed application includes:

### Services (from `app/services/services.py`)
- **OpenAI LLM Client** - For RAG generation
- **Nomic ModernBERT Embedder** - 768-dimensional embeddings
- **RAG Search Chain** - Retrieval-augmented generation
- **Document Processing Pipeline**
- **Content Cluster** with redundancy

### Schema (from `app/schemas/doc.py`)
- Document fields with metadata
- Vector embeddings for title and chunks
- HNSW index for fast similarity search
- BM25 text search on title and chunks

### Rank Profiles (from `app/profiles/`)
- `base-features` - Feature extraction
- `collect-training-data` - For model training
- `collect-second-phase` - Second-phase ranking data
- `learned-linear` - Linear model with learned weights

### Query Profiles (from `app/search/query_profiles/`)
- `hybrid` - BM25 + vector similarity
- `rag` - RAG with OpenAI LLM
- `deepresearch` - Enhanced retrieval

## Troubleshooting

**Vespa deployment takes too long:**
- First deployment downloads Docker images (~2-5 minutes)
- Subsequent deployments are faster (~30-60 seconds)

**Connection errors:**
- Ensure Docker is running
- Check Docker socket path is correct for your OS
- On macOS: `~/.docker/run/docker.sock`

**Documents not feeding:**
- Check Vespa is fully initialized (wait for "Successfully deployed" message)
- Verify the dataset file exists at `app/dataset/docs.jsonl`
- Check logs for specific error messages

**Query returns no results:**
- Ensure documents have been fed successfully
- Verify the query profile name is correct
- Check Vespa status: `curl http://localhost:8000/vespa/status`

## OpenAI Integration (Optional)

To use the RAG query profile with OpenAI:

1. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Or configure it in Vespa Cloud Secret Store (for production)

Without an API key, the RAG profile may not work, but hybrid and deepresearch profiles will function normally.

## Application Architecture

All functionality is consolidated in `main.py`:

1. **Automatic Deployment** - Vespa application is built and deployed on startup
2. **Automatic Document Feeding** - Documents from `app/dataset/docs.jsonl` are fed automatically
3. **REST API** - FastAPI endpoints for querying and manual operations
4. **Single Entry Point** - Everything managed through one application

### Code Structure

- `main.py` - Main FastAPI application with Vespa deployment and document feeding
- `app/services/services.py` - Vespa services configuration (pyvespa equivalent of services.xml)
- `app/schemas/doc.py` - Document schema definition
- `app/profiles/` - Rank profile definitions
- `app/search/query_profiles/` - Query profile configurations
- `app/dataset/docs.jsonl` - 100 sample documents

## Next Steps

- Explore the API documentation at `http://localhost:8000/docs`
- Try different query profiles with various search queries
- Add your own documents to `app/dataset/docs.jsonl` and restart the app
- Feed individual documents using the `/feed` endpoint
- Customize rank profiles for your use case
- Integrate with your application via the REST API
