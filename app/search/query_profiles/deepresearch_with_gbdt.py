from vespa.package import (
    QueryProfile,
    QueryProfileType,
    QueryTypeField,
)
from hybrid_with_gbdt import create_hybrid_with_gbdt_query_profile as create_hybrid_with_gbdt_query_profile
def create_deepresearch_with_gbdt_query_profile():
    # First, create the query profile type
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
                value="100"
            ),
            QueryTypeField(
                name="timeout",
                type="string",
                value="5s"
            )
        ]
    )
    
    # Then create the query profile
    query_profile = QueryProfile(
        id="deepresearch-with-gbdt",
        inherits=create_hybrid_with_gbdt_query_profile(),  # Inherits from another query profile
        type=query_profile_type
    )
    
    return query_profile