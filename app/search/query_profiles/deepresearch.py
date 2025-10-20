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
def create_deepresearch_query_profile():
    # Create the query profile type with deep research specific fields
    query_profile_type = QueryProfileType(
        fields=[
            QueryTypeField(
                name="yql",
                type="string",
                value="select * from %{schema} where userInput(@query) or ({label:\"title_label\", targetHits:10000}nearestNeighbor(title_embedding, embedding)) or ({label:\"chunks_label\", targetHits:10000}nearestNeighbor(chunk_embeddings, embedding))"
            ),
            QueryTypeField(
                name="hits",
                type="integer",
                value="100"  # Override hits from hybrid profile (was 10)
            ),
            QueryTypeField(
                name="timeout",
                type="string",
                value="5s"  # 5 second timeout for deep research
            )
        ]
    )
    
    # Create the query profile that inherits from hybrid
    query_profile = QueryProfile(
        id="deepresearch",
        inherits=create_hybrid_query_profile(),  # Inherit from the hybrid query profile
        type=query_profile_type
    )
    
    return query_profile