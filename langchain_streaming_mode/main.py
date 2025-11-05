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
import unicodedata
from vespa.package import Component, Parameter
logger = getLogger(__name__)
vespa_app = None
vespa_docker = None

# Set DOCKER_HOST for macOS Docker Desktop
# Docker Desktop on macOS uses ~/.docker/run/docker.sock by default
docker_socket_path = Path.home() / ".docker" / "run" / "docker.sock"
if docker_socket_path.exists() and "DOCKER_HOST" not in os.environ:
    os.environ["DOCKER_HOST"] = f"unix://{docker_socket_path}"

logger = getLogger(__name__)

def sample_pdfs():
    return [
        {
            "title": "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction",
            "url": "https://arxiv.org/pdf/2112.01488.pdf",
            "authors": "Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia",
        },
        {
            "title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
            "url": "https://arxiv.org/pdf/2004.12832.pdf",
            "authors": "Omar Khattab, Matei Zaharia",
        },
        {
            "title": "On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval",
            "url": "https://arxiv.org/pdf/2108.11480.pdf",
            "authors": "Craig Macdonald, Nicola Tonellotto",
        },
        {
            "title": "A Study on Token Pruning for ColBERT",
            "url": "https://arxiv.org/pdf/2112.06540.pdf",
            "authors": "Carlos Lassance, Maroua Maachou, Joohee Park, Stéphane Clinchant",
        },
        {
            "title": "Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval",
            "url": "https://arxiv.org/pdf/2106.11251.pdf",
            "authors": "Xiao Wang, Craig Macdonald, Nicola Tonellotto, Iadh Ounis",
        },
    ]

from vespa.package import Schema, Document, Field, FieldSet, HNSW


def create_schema():
    schema = Schema(
    name="pdf",
    mode="streaming",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary", "index"]),
            Field(name="title", type="string", indexing=["summary", "index"]),
            Field(name="url", type="string", indexing=["summary", "index"]),
            Field(name="authors", type="array<string>", indexing=["summary", "index"]),
            Field(name="page", type="int", indexing=["summary", "index"]),
            Field(
                name="metadata",
                type="map<string,string>",
                indexing=["summary", "index"],
            ),
            Field(
                name="chunks", 
                type="array<string>", 
                indexing=["summary", "index"]
            ),
            Field(
                name="embedding",
                type="tensor(chunk{}, x[384])",
                indexing=["input chunks", "embed e5", "attribute", "index"],
                ann=HNSW(distance_metric="angular"),
                is_document_field=False,
            ),
        ],
    ),
    fieldsets=[FieldSet(name="default", fields=["chunks", "title"])],
    )
    return schema

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

from vespa.package import RankProfile, Function, FirstPhaseRanking

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,  # chars, not llm tokens
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

def create_documents():
    import hashlib
    my_docs_to_feed = []
    my_docs_to_feed = []
    for pdf in sample_pdfs():
        url = pdf["url"]
        loader = PyPDFLoader(url)
        pages = loader.load_and_split()
        for index, page in enumerate(pages):
            source = page.metadata["source"]
            chunks = text_splitter.transform_documents([page])
            text_chunks = [chunk.page_content for chunk in chunks]
            text_chunks = [remove_control_characters(chunk) for chunk in text_chunks]
            page_number = index + 1
            vespa_id = f"{url}#{page_number}"
            hash_value = hashlib.sha1(vespa_id.encode()).hexdigest()
            fields = {
                "title": pdf["title"],
                "url": url,
                "page": page_number,
                "id": hash_value,
                "authors": [a.strip() for a in pdf["authors"].split(",")],
                "chunks": text_chunks,
                "metadata": page.metadata,
            }
            my_docs_to_feed.append(fields)
        return my_docs_to_feed

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

def create_rank_profile():
    rank_profile = RankProfile(
    name="hybrid",
    inputs=[("query(q)", "tensor(x[384])")],
    functions=[
        Function(
            name="similarities",
            expression="cosine_similarity(query(q), attribute(embedding),x)",
        )
    ],
    first_phase=FirstPhaseRanking(
        expression="nativeRank(title) + nativeRank(chunks) + reduce(similarities, max, chunk)",
        rank_score_drop_limit=0.0,
    ),
    match_features=[
        "closest(embedding)",
        "similarities",
        "nativeRank(chunks)",
        "nativeRank(title)",
        "elementSimilarity(chunks)",
    ],
)
    return rank_profile

def create_layered_rank_profile():
    rank_profile = RankProfile(
        name="layeredranking",  
        inputs = [("query(q)", "tensor(x[384])")],
        functions=[
            Function(
                name="my_distance",
                expression="euclidean_distance(query(q), attribute(embedding), x)",
            ),
            Function(
                name="my_distance_scores",
                expression="1 / (1+my_distance)",
            ),
            Function(
                name="my_text_scores",
                expression="elementwise(bm25(chunks), chunk, float)",
            ),
            Function(
                name="chunk_scores",
                expression="join(my_distance_scores, my_text_scores, f(a,b)(a+b))",
            ),
            Function(
                name="best_chunks",
                expression="top(3, chunk_scores)",
            ),
        ],
        first_phase="sum(chunk_scores())",
        match_features=[
            "my_distance",
            "my_distance_scores", 
            "my_text_scores",
            "chunk_scores",
            "best_chunks",
        ],
        summary_features=["best_chunks"]
    )
    return rank_profile

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vespa_app, vespa_docker
    logger.info("Starting up: Building Vespa application package...")

    try:
        schema = create_schema()
        
        schema.add_rank_profile(create_rank_profile())
        schema.add_rank_profile(create_layered_rank_profile())
        package = ApplicationPackage(
            name="langchainstreaming",
            schema=[schema],
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
from typing import Iterable

def vespa_feed(user: str) -> Iterable[dict]:
    for doc in create_documents():
        yield {"fields": doc, "id": doc["id"], "groupname": user}

@app.get("/batch-feed")    
async def batch_feed():
    from vespa.io import VespaResponse, VespaQueryResponse
    async with vespa_app.asyncio() as async_app:
        def callback(response: VespaResponse, id: str):
            if not response.is_successful():
                print(f"Document {id} failed to feed with status code {response.status_code}, url={response.url} response={response.get_json()}")
        vespa_app.feed_iterable(schema="pdf", iter=vespa_feed("jo-bergum"), namespace="personal", callback=callback)
        return {"message": "Documents fed successfully"}

from fastapi import Query
@app.get("/query")  
async def query_endpoint(q: str = Query(..., alias="query", description="Search query text")):
    response: VespaQueryResponse = vespa_app.query(
        yql="select id,title,page,chunks from pdf where userQuery() or ({targetHits:10}nearestNeighbor(embedding,q))",
        groupname="jo-bergum",
        ranking="hybrid",
        query="why is colbert effective?",
        body={
            "presentation.format.tensors": "short-value",
            "input.query(q)": 'embed(e5, "why is colbert effective?")',
        },
        timeout="2s",
    )   
    if not response.is_successful():
        return {
            "status": "error",
            "message": "Query failed",
            "status_code": response.status_code,
            "response": response.get_json()
        }
    return json.dumps(response.get_json(), indent=2)

@app.get("/query-layered")  
async def query_endpoint(q: str = Query(..., alias="query", description="Search query text")):
    response: VespaQueryResponse = vespa_app.query(
        yql="select id,title,page,chunks from pdf where userQuery() or ({targetHits:10}nearestNeighbor(embedding,q))",
        groupname="jo-bergum",
        ranking="layeredranking",
        query=q,
        body={
            "presentation.format.tensors": "short-value",
            "input.query(q)": f'embed(e5, "{q}")',
        },
        timeout="2s",
    )   
    if not response.is_successful():
        return {
            "status": "error",
            "message": "Query failed",
            "status_code": response.status_code,
            "response": response.get_json()
        }
    return json.dumps(response.get_json(), indent=2)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)