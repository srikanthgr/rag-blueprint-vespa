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


from vespa.package import ApplicationPackage
from vespa.deployment import VespaDocker
from logging import getLogger

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vespa_app, vespa_docker
    logger.info("Starting up: Building Vespa application package...")

    try:
        # Create rank profiles
        rank_profiles = [
            create_base_features_profile(),
            create_collect_training_data_profile(),
            create_collect_second_phase_profile(),
            create_learned_linear_profile(),
            # create_second_with_gbdt_profile(),  # Requires LightGBM model file
        ]

        # Create schema with rank profiles
        schema = create_docs_schema()
        schema.rank_profiles = {profile.name: profile for profile in rank_profiles}

        # Create query profiles
        query_profiles = [
            create_hybrid_query_profile(),
            create_rag_query_profile(),
            create_deepresearch_query_profile(),
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
            application_package=app_package
        )
        logger.info(f"Successfully deployed Vespa application at {vespa_app.url}")

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
        "status": "ready" if vespa_app else "initializing"
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
async def feed_document(document: dict):
    """
    Feed a document to Vespa

    Expected document format:
    {
        "id": "doc-id",
        "fields": {
            "id": "doc-id",
            "title": "Document title",
            "text": "Document content",
            "created_timestamp": 1234567890,
            "modified_timestamp": 1234567890,
            "last_opened_timestamp": 1234567890,
            "open_count": 0,
            "favorite": false
        }
    }
    """
    if vespa_app is None:
        return {"error": "Vespa instance not initialized"}, 503

    try:
        response = vespa_app.feed_data_point(
            schema="doc",
            data_id=document["id"],
            fields=document["fields"]
        )
        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Error feeding document: {e}")
        return {"status": "error", "error": str(e)}

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
            "health": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
