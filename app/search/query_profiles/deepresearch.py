from vespa.package import QueryProfile, QueryField

def create_deepresearch_query_profile():
    """
    Create deep research query profile for comprehensive document retrieval.

    This profile is optimized for research tasks with:
    - Higher targetHits for more comprehensive recall
    - More results returned (100 hits)
    - Longer timeout for thorough search
    """

    # Create the query profile with all fields
    query_profile = QueryProfile(
        fields=[
            # Schema and ranking features (from hybrid)
            QueryField(name="schema", value="doc"),
            QueryField(name="ranking.features.query(embedding)", value="embed(@query)"),
            QueryField(name="ranking.features.query(float_embedding)", value="embed(@query)"),
            QueryField(name="ranking.features.query(intercept)", value="-7.798639"),
            QueryField(name="ranking.features.query(avg_top_3_chunk_sim_scores_param)", value="13.383840"),
            QueryField(name="ranking.features.query(avg_top_3_chunk_text_scores_param)", value="0.203145"),
            QueryField(name="ranking.features.query(bm25_chunks_param)", value="0.159914"),
            QueryField(name="ranking.features.query(bm25_title_param)", value="0.191867"),
            QueryField(name="ranking.features.query(max_chunk_sim_scores_param)", value="10.067169"),
            QueryField(name="ranking.features.query(max_chunk_text_scores_param)", value="0.153392"),
            QueryField(name="ranking.profile", value="learned-linear"),
            QueryField(name="presentation.summary", value="top_3_chunks"),
            # Deep research overrides
            QueryField(
                name="yql",
                value="select * from doc where userInput(@query) or ({label:\"title_label\", targetHits:10000}nearestNeighbor(title_embedding, embedding)) or ({label:\"chunks_label\", targetHits:10000}nearestNeighbor(chunk_embeddings, embedding))"
            ),
            QueryField(name="hits", value=100),  # More results for comprehensive research
            QueryField(name="timeout", value="5s"),  # Longer timeout
        ]
    )

    return query_profile