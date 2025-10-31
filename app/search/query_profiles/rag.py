from vespa.package import QueryProfile, QueryField

def create_rag_query_profile():
    """
    Create RAG query profile for retrieval-augmented generation.

    This profile extends hybrid search with OpenAI LLM integration for generation.
    """

    # Create the query profile with all fields (including inherited from hybrid)
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
            QueryField(
                name="yql",
                value="select * from doc where userInput(@query) or ({label:\"title_label\", targetHits:100}nearestNeighbor(title_embedding, embedding)) or ({label:\"chunks_label\", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))"
            ),
            QueryField(name="ranking.profile", value="learned-linear"),
            QueryField(name="presentation.summary", value="top_3_chunks"),
            # RAG-specific overrides
            QueryField(name="hits", value=50),  # More hits for RAG context
            QueryField(name="searchChain", value="openai"),  # Use OpenAI chain
            QueryField(name="presentation.format", value="sse"),  # Streaming response
        ]
    )

    return query_profile