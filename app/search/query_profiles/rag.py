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
    QueryProfile,
    QueryProfileType,
    QueryTypeField,
)
from hybrid import create_hybrid_query_profile as create_hybrid_query_profile
def create_rag_query_profile():
    # Create the query profile type with RAG-specific fields
    query_profile_type = QueryProfileType(
        fields=[
            QueryTypeField(
                name="hits",
                type="integer",
                value="50"  # Override hits from hybrid profile (was 10)
            ),
            QueryTypeField(
                name="searchChain",
                type="string",
                value="openai"  # Use OpenAI search chain for RAG
            ),
            QueryTypeField(
                name="presentation.format",
                type="string",
                value="sse"  # Server-Sent Events format for streaming
            )
        ]
    )
    
    # Create the query profile that inherits from hybrid
    query_profile = QueryProfile(
        id="rag",
        inherits=create_hybrid_query_profile(),  # Inherit from the hybrid query profile
        type=query_profile_type
    )
    
    return query_profile