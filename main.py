"""
FastAPI application with Vespa RAG deployment

This application deploys a complete Vespa RAG system with:
- OpenAI LLM integration
- Nomic ModernBERT embeddings
- Hybrid search with BM25 and vector similarity
- Multiple query profiles (hybrid, RAG, deep research)
- Rank profiles with learned linear models

The Vespa instance is available globally via vespa_app for document feeding and querying.
"""

from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set Docker socket path for macOS
os.environ['DOCKER_HOST'] = 'unix://' + os.path.expanduser('~/.docker/run/docker.sock')

from vespa.package import ApplicationPackage
from vespa.deployment import VespaDocker
from logging import getLogger
import json
from typing import List, Dict

from app.services import create_services
from app.profiles import (
    create_base_features_profile,
    create_collect_training_data_profile,
    create_collect_second_phase_profile,
    create_learned_linear_profile,
)
from app.search.query_profiles import (
    create_rag_query_profile,
    create_hybrid_query_profile,
    create_deepresearch_query_profile,
)
from app.schemas import create_docs_schema

logger = getLogger(__name__)

# Global variables to store Vespa instances
vespa_app = None
vespa_docker = None

def read_jsonl_documents(file_path: str) -> List[Dict]:
    """
    Read documents from JSONL file.

    Expected format per line:
    {
        "put": "id:doc:doc::ID",
        "fields": {
            "id": "ID",
            "title": "...",
            "text": "...",
            ...
        }
    }
    """
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line.strip())
                    if 'fields' in doc:
                        documents.append(doc)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
        logger.info(f"Successfully read {len(documents)} documents from {file_path}")
        return documents
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading JSONL file: {e}")
        return []


def feed_documents_to_vespa(documents: List[Dict]) -> Dict:
    """Feed documents to Vespa and return summary"""
    if vespa_app is None:
        return {"error": "Vespa instance not initialized"}

    success_count = 0
    failed_count = 0
    errors = []

    logger.info(f"Starting to feed {len(documents)} documents to Vespa...")

    for i, doc in enumerate(documents, 1):
        try:
            fields = doc.get("fields", {})
            doc_id = fields.get("id")

            if not doc_id:
                logger.warning(f"Document {i} missing 'id' field, skipping")
                failed_count += 1
                errors.append(f"Document {i}: Missing 'id' field")
                continue

            response = vespa_app.feed_data_point(
                schema="doc",
                data_id=str(doc_id),
                fields=fields
            )

            success_count += 1
            if i % 10 == 0:  # Log progress every 10 documents
                logger.info(f"Progress: {i}/{len(documents)} documents processed")

        except Exception as e:
            failed_count += 1
            error_msg = f"Document {doc_id if 'doc_id' in locals() else i}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)

    logger.info(f"Batch feed completed: {success_count} successful, {failed_count} failed")

    return {
        "total_documents": len(documents),
        "successful": success_count,
        "failed": failed_count,
        "errors": errors[:10] if errors else []
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vespa_app, vespa_docker
    logger.info("Starting up: Building Vespa application package...")

    try:
        # Create rank profiles (commented out temporarily to test basic deployment)
        # rank_profiles = [
        #     create_base_features_profile(),
        #     create_collect_training_data_profile(),
        #     create_collect_second_phase_profile(),
        #     create_learned_linear_profile(),
        #     # create_second_with_gbdt_profile(),  # Requires LightGBM model file
        # ]

        # Create schema (without rank profiles for now)
        schema = create_docs_schema()
        # schema.rank_profiles = {profile.name: profile for profile in rank_profiles}

        # Create query profiles
        query_profiles = [
            create_hybrid_query_profile(),
            # create_rag_query_profile(),
            # create_deepresearch_query_profile(),
        ]

        # Build application package with all components
        app_package = ApplicationPackage(
            name="rag",
            schema=[schema],
            services_config=create_services(),
            query_profile=query_profiles,
        )

        logger.info("Deploying Vespa application to Docker...")
        vespa_docker = VespaDocker()
        vespa_app = vespa_docker.deploy(
            application_package=app_package,
            max_wait_configserver=300  # Wait up to 5 minutes for config server
        )
        logger.info(f"Successfully deployed Vespa application at {vespa_app.url}")
        logger.info("Vespa is ready! Use POST /batch-feed to load documents.")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    yield

    logger.info("Shutting down: Cleaning up Vespa Docker container...")
    try:
        if vespa_docker and vespa_docker.container:
            vespa_docker.container.stop()
            vespa_docker.container.remove()
            logger.info("Vespa Docker container stopped and removed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        vespa_app = None
        vespa_docker = None
    logger.info("Cleanup completed")

app = FastAPI(
    title="RAG Blueprint with Vespa",
    description="FastAPI application with Vespa RAG deployment for document search and retrieval",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/")
def root_controller():
    """Root endpoint - health check"""
    return {
        "message": "RAG Blueprint with Vespa is running!",
        "vespa_url": vespa_app.url if vespa_app else None,
        "status": "ready" if vespa_app else "initializing",
        "next_step": "Call POST /batch-feed to load 100 sample documents" if vespa_app else "Waiting for Vespa to initialize..."
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    if vespa_app is None:
        return {"status": "unhealthy", "message": "Vespa instance not initialized"}

    try:
        # Try to get application status
        return {
            "status": "healthy",
            "vespa_url": vespa_app.url,
            "application": "rag"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/feed")
async def feed_document():
    """
    Feed all documents from app/dataset/docs.jsonl to Vespa.

    This endpoint loads all documents from the dataset into Vespa.
    No request body is required - it automatically reads from the docs.jsonl file.

    Returns a summary of successful and failed document feeds.
    """
    if vespa_app is None:
        return {"error": "Vespa instance not initialized"}, 503

    # Path to the dataset
    dataset_path = Path(__file__).parent / "app" / "dataset" / "docs.jsonl"

    if not dataset_path.exists():
        return {
            "status": "error",
            "error": f"Dataset file not found at {dataset_path}"
        }

    logger.info(f"Feed requested from {dataset_path}")

    # Read documents
    documents = read_jsonl_documents(str(dataset_path))

    if not documents:
        return {
            "status": "error",
            "error": "No documents found in file"
        }

    # Feed documents using the helper function
    result = feed_documents_to_vespa(documents)

    return {
        "status": "completed",
        **result
    }

@app.post("/query")
async def query_documents(query: dict):
    """
    Query Vespa for documents

    Expected query format:
    {
        "query": "search query text",
        "query_profile": "hybrid",  # or "rag", "deepresearch"
        "hits": 10
    }
    """
    if vespa_app is None:
        return {"error": "Vespa instance not initialized"}, 503

    try:
        response = vespa_app.query(
            query=query.get("query", ""),
            query_profile=query.get("query_profile", "hybrid"),
            hits=query.get("hits", 10)
        )
        return {"status": "success", "response": response.json}
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/vespa/status")
def vespa_status():
    """Get Vespa application status"""
    if vespa_app is None:
        return {"status": "not_initialized"}

    return {
        "status": "running",
        "url": vespa_app.url,
        "application_name": "rag",
        "endpoints": {
            "feed": "/feed",
            "query": "/query",
            "health": "/health",
            "batch_feed": "/batch-feed"
        }
    }

@app.post("/batch-feed")
async def batch_feed_documents():
    """
    Feed all documents from app/dataset/docs.jsonl to Vespa.

    This endpoint loads all 100 documents from the dataset into Vespa.
    Call this after the application starts to populate the search index.

    Returns a summary of successful and failed document feeds.
    """
    if vespa_app is None:
        return {"error": "Vespa instance not initialized"}, 503

    # Path to the dataset
    dataset_path = Path(__file__).parent / "app" / "dataset" / "docs.jsonl"

    if not dataset_path.exists():
        return {
            "status": "error",
            "error": f"Dataset file not found at {dataset_path}"
        }

    logger.info(f"Manual batch feed requested from {dataset_path}")

    # Read documents
    documents = read_jsonl_documents(str(dataset_path))

    if not documents:
        return {
            "status": "error",
            "error": "No documents found in file"
        }

    # Feed documents using the helper function
    result = feed_documents_to_vespa(documents)

    return {
        "status": "completed",
        **result
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
