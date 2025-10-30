from vespa.package import (
    QueryProfile,
    QueryProfileType,
    QueryTypeField,
)
from hybrid_with_gbdt import create_hybrid_with_gbdt_query_profile as create_hybrid_with_gbdt_query_profile
def create_rag_with_gbdt_query_profile():
    # Create the query profile type with RAG-specific fields
    query_profile_type = QueryProfileType(
        fields=[
            QueryTypeField(
                name="hits",
                type="integer",
                value="50"  # Override hits from hybrid-with-gbdt profile (was 20)
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
    
    # Create the query profile that inherits from hybrid-with-gbdt
    query_profile = QueryProfile(
        id="rag-with-gbdt",
        inherits=create_hybrid_with_gbdt_query_profile(),  # Inherit from the hybrid-with-gbdt query profile
        type=query_profile_type
    )
    
    return query_profile