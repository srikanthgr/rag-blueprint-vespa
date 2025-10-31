from vespa.package import QueryProfile, QueryField

def create_rag_with_gbdt_query_profile():
    """
    Create RAG query profile with GBDT ranking.

    Combines RAG generation with GBDT second-phase ranking.
    """

    # Create the query profile with RAG + GBDT
    query_profile = QueryProfile(
        fields=[
            # Schema and ranking features
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
            QueryField(name="presentation.summary", value="top_3_chunks"),
            # RAG + GBDT specific
            QueryField(name="hits", value=50),
            QueryField(name="ranking.profile", value="second-with-gbdt"),
            QueryField(name="searchChain", value="openai"),
            QueryField(name="presentation.format", value="sse"),
        ]
    )

    return query_profile