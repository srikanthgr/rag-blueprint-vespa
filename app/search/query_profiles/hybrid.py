from vespa.package import QueryProfile, QueryField

def create_hybrid_query_profile():
    """
    Create hybrid query profile for BM25 + vector similarity search.

    This profile combines:
    - BM25 text search on title and chunks
    - Vector similarity search on title and chunk embeddings
    - Learned linear ranking model
    """

    # Create the query profile with fields (simplified for basic deployment)
    query_profile = QueryProfile(
        fields=[
            QueryField(name="schema", value="doc"),
            QueryField(
                name="yql",
                value="select * from doc where userInput(@query) or ({targetHits:100}nearestNeighbor(title_embedding, float_embedding)) or ({targetHits:100}nearestNeighbor(chunk_embeddings, float_embedding))"
            ),
            QueryField(name="hits", value=10),
            QueryField(name="ranking.features.query(float_embedding)", value="embed(@query)"),
        ]
    )

    return query_profile