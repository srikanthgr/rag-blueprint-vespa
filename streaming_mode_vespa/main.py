from fastapi import FastAPI, Query
from contextlib import asynccontextmanager
from logging import getLogger
from vespa.package import ApplicationPackage
from vespa.deployment import VespaDocker
import uvicorn
import os
from pathlib import Path

# Set DOCKER_HOST for macOS Docker Desktop
# Docker Desktop on macOS uses ~/.docker/run/docker.sock by default
docker_socket_path = Path.home() / ".docker" / "run" / "docker.sock"
if docker_socket_path.exists() and "DOCKER_HOST" not in os.environ:
    os.environ["DOCKER_HOST"] = f"unix://{docker_socket_path}"

logger = getLogger(__name__)

# Global variables to store Vespa instances
vespa_app = None
vespa_docker = None

from typing import List, Union, Optional

from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

class PersonalAssistantVespaRetriever(BaseRetriever):
    def __init__(
        self, 
        vespa_app: Vespa, 
        user: str, 
        hits: int = 5,
        vespa_rank_profile: str = "default",
        vespa_score_cutoff: float = 0.70,
        sources: List[str] = ["mail"],
        fields: List[str] = ["subject", "body"],
        callback_manager: Optional[CallbackManager] = None,

    ) -> None:

        self.vespa_app = vespa_app
        self.user = user
        self.hits = hits
        self.vespa_rank_profile = vespa_rank_profile
        self.vespa_score_cutoff = vespa_score_cutoff
        self.sources = sources
        self.fields = fields
        self.callback_manager = callback_manager

    def _retrieve(self, query: Union[str, QueryBundle]) -> List[NodeWithScore]:
        """Retrieve documents from Vespa application."""
        if isinstance(query, QueryBundle):
            query = query.query_str

        if self.vespa_rank_profile == "default":
            yql: str = f"select {self.summary_fields} from mail where userQuery()"
        else:
            yql = f"select {self.summary_fields} from sources {self.sources} where {{targetHits:10}}nearestNeighbor(embedding,q) or userQuery()"
        vespa_body_request = {
            "yql": yql,
            "query": query,
            "hits": self.hits,
            "ranking.profile": self.vespa_rank_profile,
            "timeout": "2s",
            "input.query(threshold)": self.vespa_score_cutoff,
        }
        if self.vespa_rank_profile != "default":
            vespa_body_request["input.query(q)"] = f'embed(e5, "{query}")'

        with self.app.syncio(connections=1) as session:
            response: VespaQueryResponse = session.query(
                body=vespa_body_request, groupname=self.user
            )
            if not response.is_successful():
                raise ValueError(
                    f"Query request failed: {response.status_code}, response payload: {response.get_json()}"
                )

        nodes: List[NodeWithScore] = []
        for hit in response.hits:
            response_fields: dict = hit.get("fields", {})
            text: str = ""
            for field in response_fields.keys():
                if isinstance(response_fields[field], str) and field in self.fields:
                    text += response_fields[field] + " "
            id = hit["id"]
            #
            doc = TextNode(
                id_=id,
                text=text,
                metadata=response_fields,
            )
            nodes.append(NodeWithScore(node=doc, score=hit["relevance"]))
        return nodes

def synthetic_mail_data_generator() -> List[dict]:
    synthetic_mails = [
        {
            "id": 1,
            "groupname": "bergum@vespa.ai",
            "fields": {
                "subject": "LlamaIndex news, 2023-11-14",
                "to": "bergum@vespa.ai",
                "body": """Hello Llama Friends 🦙 LlamaIndex is 1 year old this week! 🎉 To celebrate, we're taking a stroll down memory 
                    lane on our blog with twelve milestones from our first year. Be sure to check it out.""",
                "from": "news@llamaindex.ai",
                "display_date": "2023-11-15T09:00:00Z",
            },
        },
        {
            "id": 2,
            "groupname": "bergum@vespa.ai",
            "fields": {
                "subject": "Dentist Appointment Reminder",
                "to": "bergum@vespa.ai",
                "body": "Dear Jo Kristian ,\nThis is a reminder for your upcoming dentist appointment on 2023-12-04 at 09:30. Please arrive 15 minutes early.\nBest regards,\nDr. Dentist",
                "from": "dentist@dentist.no",
                "display_date": "2023-11-15T15:30:00Z",
            },
        },
        {
            "id": 1,
            "groupname": "giraffe@wildlife.ai",
            "fields": {
                "subject": "Wildlife Update: Giraffe Edition",
                "to": "giraffe@wildlife.ai",
                "body": "Dear Wildlife Enthusiasts 🦒, We're thrilled to share the latest insights into giraffe behavior in the wild. Join us on an adventure as we explore their natural habitat and learn more about these majestic creatures.",
                "from": "updates@wildlife.ai",
                "display_date": "2023-11-12T14:30:00Z",
            },
        },
        {
            "id": 1,
            "groupname": "penguin@antarctica.ai",
            "fields": {
                "subject": "Antarctica Expedition: Penguin Chronicles",
                "to": "penguin@antarctica.ai",
                "body": "Greetings Explorers 🐧, Our team is embarking on an exciting expedition to Antarctica to study penguin colonies. Stay tuned for live updates and behind-the-scenes footage as we dive into the world of these fascinating birds.",
                "from": "expedition@antarctica.ai",
                "display_date": "2023-11-11T11:45:00Z",
            },
        },
        {
            "id": 1,
            "groupname": "space@exploration.ai",
            "fields": {
                "subject": "Space Exploration News: November Edition",
                "to": "space@exploration.ai",
                "body": "Hello Space Enthusiasts 🚀, Join us as we highlight the latest discoveries and breakthroughs in space exploration. From distant galaxies to new technologies, there's a lot to explore!",
                "from": "news@exploration.ai",
                "display_date": "2023-11-01T16:20:00Z",
            },
        },
        {
            "id": 1,
            "groupname": "ocean@discovery.ai",
            "fields": {
                "subject": "Ocean Discovery: Hidden Treasures Unveiled",
                "to": "ocean@discovery.ai",
                "body": "Dear Ocean Explorers 🌊, Dive deep into the secrets of the ocean with our latest discoveries. From undiscovered species to underwater landscapes, our team is uncovering the wonders of the deep blue.",
                "from": "discovery@ocean.ai",
                "display_date": "2023-10-01T10:15:00Z",
            },
        },
    ]
    for mail in synthetic_mails:
        yield mail


def synthetic_calendar_data_generator() -> List[dict]:
    calendar_data = [
        {
            "id": 1,
            "groupname": "bergum@vespa.ai",
            "fields": {
                "subject": "Dentist Appointment",
                "to": "bergum@vespa.ai",
                "body": "Dentist appointment at 2023-12-04 at 09:30 - 1 hour duration",
                "from": "dentist@dentist.no",
                "display_date": "2023-11-15T15:30:00Z",
                "duration": 60,
            },
        },
        {
            "id": 2,
            "groupname": "bergum@vespa.ai",
            "fields": {
                "subject": "Public Cloud Platform Events",
                "to": "bergum@vespa.ai",
                "body": "The cloud team continues to push new features and improvements to the platform. Join us for a live demo of the latest updates",
                "from": "public-cloud-platform-events",
                "display_date": "2023-11-21T09:30:00Z",
                "duration": 60,
            },
        },
    ]
    for event in calendar_data:
        yield event

from vespa.package import Schema, Document, Field, FieldSet, HNSW

def create_mails_schema():
    mail_schema = Schema(
        name="mail",
        mode="streaming",
        document=Document(
            fields=[
                Field(name="id", type="string", indexing=["summary", "index"]),
                Field(name="subject", type="string", indexing=["summary", "index"]),
                Field(name="to", type="string", indexing=["summary", "index"]),
                Field(name="from", type="string", indexing=["summary", "index"]),
                Field(name="body", type="string", indexing=["summary", "index"]),
                Field(name="display_date", type="string", indexing=["summary"]),
                Field(
                    name="timestamp",
                    type="long",
                    indexing=[
                        "input display_date",
                        "to_epoch_second",
                        "summary",
                        "attribute",
                    ],
                    is_document_field=False,
                ),
                Field(
                    name="embedding",
                    type="tensor<bfloat16>(x[384])",
                    indexing=[
                        'input subject ." ". input body',
                        "embed e5",
                        "attribute",
                        "index",
                    ],
                    ann=HNSW(distance_metric="angular"),
                    is_document_field=False,
                ),
            ],
        ),
        fieldsets=[FieldSet(name="default", fields=["subject", "body", "to", "from"])],
    )
    return mail_schema

def create_calendar_schema():
    calendar_schema = Schema(
        name="calendar",
        inherits="mail",
        mode="streaming",
        document=Document(
            inherits="mail",
            fields=[
                Field(name="duration", type="int", indexing=["summary", "index"]),
                Field(name="guests", type="array<string>", indexing=["summary", "index"]),
                Field(name="location", type="string", indexing=["summary", "index"]),
                Field(name="url", type="string", indexing=["summary", "index"]),
                Field(name="address", type="string", indexing=["summary", "index"]),
            ],
        ),
    )
    return calendar_schema

from vespa.package import RankProfile, Function, GlobalPhaseRanking, FirstPhaseRanking, SecondPhaseRanking

keywords_and_freshness = RankProfile(
    name="default",
    functions=[
        Function(
            name="my_function",
            expression="nativeRank(subject) + nativeRank(body) + freshness(timestamp)",
        )
    ],
    first_phase=FirstPhaseRanking(expression="my_function", rank_score_drop_limit=0.02),
    match_features=[
        "nativeRank(subject)",
        "nativeRank(body)",
        "my_function",
        "freshness(timestamp)",
    ],
)

semantic = RankProfile(
    name="semantic",
    functions=[
        Function(name="cosine", expression="max(0,cos(distance(field, embedding)))")
    ],
    inputs=[("query(q)", "tensor<float>(x[384])"), ("query(threshold)", "", "0.75")],
    first_phase=FirstPhaseRanking(
        expression="if(cosine > query(threshold), cosine, -1)",
        rank_score_drop_limit=0.1,
    ),
    match_features=[
        "cosine",
        "freshness(timestamp)",
        "distance(field, embedding)",
        "query(threshold)",
    ],
)

fusion = RankProfile(
    name="fusion",
    inherits="semantic",
    functions=[
        Function(
            name="keywords_and_freshness",
            expression=" nativeRank(subject) + nativeRank(body) + freshness(timestamp)",
        ),
        Function(name="semantic", expression="cos(distance(field,embedding))"),
    ],
    inputs=[("query(q)", "tensor<float>(x[384])"), ("query(threshold)", "", "0.75")],
    first_phase=FirstPhaseRanking(
        expression="if(cosine > query(threshold), cosine, -1)",
        rank_score_drop_limit=0.1,
    ),
    match_features=[
        "nativeRank(subject)",
        "keywords_and_freshness",
        "freshness(timestamp)",
        "cosine",
        "query(threshold)",
    ],
    global_phase=GlobalPhaseRanking(
        rerank_count=1000,
        expression="reciprocal_rank_fusion(semantic, keywords_and_freshness)",
    ),
)

from vespa.package import ApplicationPackage, Component, Parameter, ServicesConfiguration
from vespa.configuration.services import (
    services,
    container,
    component as component_config,
    document_api,
    document_processing,
    search,
    nodes,
    node,
    content,
    min_redundancy,
    documents,
    document,
    transformer_model,
    tokenizer_model,
    prepend,
    query,
    max_tokens,
)

vespa_app_name = "assistant"
vespa_application_package = ApplicationPackage(
    name=vespa_app_name,
    schema=[create_mails_schema(), create_calendar_schema()],
    components=[
        Component(
            id="e5",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    name="transformer-model",
                    args={
                        "url": "https://github.com/vespa-engine/sample-apps/raw/master/examples/model-exporting/model/e5-small-v2-int8.onnx"
                    },
                ),
                Parameter(
                    name="tokenizer-model",
                    args={
                        "url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/examples/model-exporting/model/tokenizer.json"
                    },
                ),
                Parameter(
                    name="prepend",
                    args={},
                    children=[
                        Parameter(name="query", args={}, children="query: "),
                        Parameter(name="document", args={}, children="passage: "),
                    ],
                ),
            ],
        )
    ],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vespa_app, vespa_docker
    logger.info("Starting up: Building Vespa application package...")

    try:

        mail_schema = create_mails_schema()
        calendar_schema = create_calendar_schema()

        mail_schema.add_rank_profile(keywords_and_freshness)        
        mail_schema.add_rank_profile(semantic)
        mail_schema.add_rank_profile(fusion)

        calendar_schema.add_rank_profile(keywords_and_freshness)
        calendar_schema.add_rank_profile(semantic)
        calendar_schema.add_rank_profile(fusion)
        
        # Create services configuration with the e5 embedder component
        # This is required for 'embed e5' in schema fields to work
        services_config = ServicesConfiguration(
            application_name="rag",
            services_config=services(
                container(
                    document_processing(),
                    document_api(),
                    # E5 embedder component for generating embeddings
                    component_config(
                        transformer_model(
                            url="https://github.com/vespa-engine/sample-apps/raw/master/examples/model-exporting/model/e5-small-v2-int8.onnx"
                        ),
                        tokenizer_model(
                            url="https://raw.githubusercontent.com/vespa-engine/sample-apps/master/examples/model-exporting/model/tokenizer.json"
                        ),
                        prepend(
                            query("query: "),
                            document("passage: ")
                        ),
                        id="e5",
                        type_="hugging-face-embedder"
                    ),
                    search(),
                    nodes(node(hostalias="node1")),
                    id="default",
                    version="1.0"
                ),
                content(
                    min_redundancy("1"),  # Minimum redundancy required for content cluster
                    documents(
                        document(type_="mail", mode="index"),
                        document(type_="calendar", mode="index")
                    ),
                    nodes(node(hostalias="node1", distribution_key="0")),
                    id="rag_content",  # Match previous deployment cluster ID
                    version="1.0"
                ),
                version="1.0"
            )
        )
        
        # Build application package with services configuration
        app_package = ApplicationPackage(
            name="rag",
            schema=[mail_schema, calendar_schema],
            services_config=services_config,  # Include services with component configuration
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

from vespa.io import VespaResponse
def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Error {response.url} : {response.get_json()}")
    else:
        print(f"Success {response.url}")

@app.get("/batch-feed")
async def batch_feed():
    results = {"mail": [], "calendar": []}

    async with vespa_app.asyncio() as async_app:
        # Feed mail documents
        for doc in synthetic_mail_data_generator():
            try:
                response = await async_app.feed_data_point(
                    schema=doc.get("schema", "mail"),
                    data_id=str(doc["id"]),
                    fields=doc["fields"],
                    namespace="assitant",
                    group=doc.get("groupname")
                )
                results["mail"].append({"id": doc["id"], "status": "success" if response.is_successful() else "failed"})
            except Exception as e:
                results["mail"].append({"id": doc["id"], "status": "error", "message": str(e)})

        # Feed calendar documents
        for doc in synthetic_calendar_data_generator():
            try:
                response = await async_app.feed_data_point(
                    schema=doc.get("schema", "calendar"),
                    data_id=str(doc["id"]),
                    fields=doc["fields"],
                    namespace="assitant",
                    group=doc.get("groupname")
                )
                results["calendar"].append({"id": doc["id"], "status": "success" if response.is_successful() else "failed"})
            except Exception as e:
                results["calendar"].append({"id": doc["id"], "status": "error", "message": str(e)})

    return {"status": "success", "message": "Documents fed successfully", "results": results}

@app.get("/query")
async def query_endpoint(
    q: str = Query(..., alias="query", description="Search query text"),
    ranking: str = Query("default", description="Ranking profile: default, semantic, or fusion")
):
    from vespa.io import VespaQueryResponse
    import json

    # Build the query body for semantic search
    # Note: 'from' is a reserved keyword in YQL, so we use * to select all fields
    body = {
        "yql": "select * from mail where {targetHits:10}nearestNeighbor(embedding,q)",
        "input.query(q)": f'embed(e5, "{q}")',
        "ranking": ranking,
        "timeout": "2s"
    }

    # Use VespaAsync context manager for proper async query execution
    async with vespa_app.asyncio() as async_app:
        response: VespaQueryResponse = await async_app.query(
            body=body,
            groupname="bergum@vespa.ai"
        )

    if not response.is_successful():
        return {
            "status": "error",
            "message": "Query failed",
            "status_code": response.status_code,
            "response": response.get_json()
        }

    return json.dumps(response.get_json(), indent=2)

@app.get("/personal-assistant")
async def personal_assistant(query: str):
    retriever = PersonalAssistantVespaRetriever(
    vespa_app=vespa_app, user="bergum@vespa.ai", vespa_rank_profile="default"
    )
    results = retriever.retrieve(query)
    return {
        "status": "success",
        "message": "Query successful",
        "results": results
    }

@app.get("/personal-assistant-async")
async def personal_assistant_async(query: str):
    retriever = PersonalAssistantVespaRetriever(
        vespa_app=vespa_app,
        user="bergum@vespa.ai",
        vespa_rank_profile="semantic",
        vespa_score_cutoff=0.6,
        hits=20,
    )
    results = retriever.retrieve(query)
    return {
        "status": "success",
        "message": "Query successful",
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)