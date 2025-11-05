from fastapi import FastAPI
from contextlib import asynccontextmanager
from logging import getLogger
from vespa.package import ApplicationPackage
from vespa.deployment import VespaDocker
from vespa.io import VespaQueryResponse
import json
from typing import List, Dict
from pathlib import Path
import os

logger = getLogger(__name__)
vespa_app = None
vespa_docker = None

# Set DOCKER_HOST for macOS Docker Desktop
# Docker Desktop on macOS uses ~/.docker/run/docker.sock by default
docker_socket_path = Path.home() / ".docker" / "run" / "docker.sock"
if docker_socket_path.exists() and "DOCKER_HOST" not in os.environ:
    os.environ["DOCKER_HOST"] = f"unix://{docker_socket_path}"

logger = getLogger(__name__)


from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Component,
    Parameter,
    FieldSet,
    GlobalPhaseRanking,
    Function,
)

def field_sets():
    fieldsets = [
        FieldSet(name="default", fields=["title", "body"])
    ]
    return fieldsets

def rank_profiles():
    rank_profiles = [
        RankProfile(
            name="bm25",
            inputs = [("query(q)", "tensor<float>(x[384])")],
            functions = [
                Function(
                    name="bm25sum", 
                    expression="bm25(title) + bm25(body)"
                )
            ],
            first_phase = "bm25sum",
        ),
        RankProfile(
            name= "semantic",
            inputs=[("query(q)", "tensor<float>(x[384])")],
            first_phase = "closeness(field, embedding)"
        ),
        RankProfile(
                    name="fusion",
                    inherits="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="closeness(field, embedding)",
                    global_phase=GlobalPhaseRanking(
                        expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))",
                        rerank_count=1000,
                    )
                )
    ]
    return rank_profiles

def create_component_config():
    components=[
        Component(
            id="e5",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    "transformer-model",
                    {
                        "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/e5-small-v2-int8.onnx"
                    },
                ),
                Parameter(
                    "tokenizer-model",
                    {
                        "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/tokenizer.json"
                    },
                ),
            ],
        )
    ]
    return components

def create_documents():
    documents = Document(
            fields = [
                Field(name= "id", type="string", indexing=["summary"]),
                Field(name="title", type="string", indexing=["summary", "index"], index="enable-bm25"),
                Field(name="body", type="string", indexing=["summary", "index"], index="enable-bm25"),
                Field(
                        name="embedding",
                        type="tensor<float>(x[384])",
                        indexing= ['input title . " " . input body', "embed e5", "attribute", "index"],
                        ann=HNSW(distance_metric="angular"),
                        is_document_field=False
                )
            ]
        )
    return documents

def create_schema():
    schema = Schema(
        name="doc",
        document = create_documents(),
        fieldsets = field_sets(),
        rank_profiles = rank_profiles()
    )
    return schema

import pandas as pd
def display_hits_as_df(response: VespaQueryResponse, fields) -> pd.DataFrame:
    records = []
    for hit in response.hits:
        record = {}
        for field in fields:
            record[field] = hit["fields"][field]
        records.append(record)
    return pd.DataFrame(records)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vespa_app, vespa_docker
    logger.info("Starting up: Building Vespa application package...")

    try:
        
        package = ApplicationPackage(
            name="hybridsearch",
            schema=[create_schema()],
            components=create_component_config(),
        )
        logger.info("Deploying Vespa application to Docker...")
        vespa_docker = VespaDocker()
        vespa_app = vespa_docker.deploy(
            application_package=package,
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

@app.post("/batch-feed")
async def batch_feed(documents: List[Dict]):
    from datasets import load_dataset

    dataset = load_dataset("BeIR/nfcorpus", "corpus", split="corpus", streaming=True)
    vespa_feed = dataset.map(
        lambda x: {
            "id": x["_id"],
            "fields": {"title": x["title"], "body": x["text"], "id": x["_id"]},
        }
    )

    from vespa.io import VespaResponse, VespaQueryResponse

    def callback(response: VespaResponse, id: str):
        if not response.is_successful():
            print(f"Error when feeding document {id}: {response.get_json()}")


    vespa_app.feed_iterable(vespa_feed, schema="doc", namespace="tutorial", callback=callback)
    return {"message": "Documents fed successfully"}
