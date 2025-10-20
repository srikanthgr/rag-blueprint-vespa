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

def create_hybrid_query_profile():
    # Create the query profile type with all necessary fields
    query_profile_type = QueryProfileType(
        fields=[
            QueryTypeField(
                name="schema",
                type="string",
                value="doc"
            ),
            QueryTypeField(
                name="ranking.features.query(embedding)",
                type="string",
                value="embed(@query)"
            ),
            QueryTypeField(
                name="ranking.features.query(float_embedding)",
                type="string",
                value="embed(@query)"
            ),
            QueryTypeField(
                name="ranking.features.query(intercept)",
                type="string",
                value="-7.798639"
            ),
            QueryTypeField(
                name="ranking.features.query(avg_top_3_chunk_sim_scores_param)",
                type="string",
                value="13.383840"
            ),
            QueryTypeField(
                name="ranking.features.query(avg_top_3_chunk_text_scores_param)",
                type="string",
                value="0.203145"
            ),
            QueryTypeField(
                name="ranking.features.query(bm25_chunks_param)",
                type="string",
                value="0.159914"
            ),
            QueryTypeField(
                name="ranking.features.query(bm25_title_param)",
                type="string",
                value="0.191867"
            ),
            QueryTypeField(
                name="ranking.features.query(max_chunk_sim_scores_param)",
                type="string",
                value="10.067169"
            ),
            QueryTypeField(
                name="ranking.features.query(max_chunk_text_scores_param)",
                type="string",
                value="0.153392"
            ),
            QueryTypeField(
                name="yql",
                type="string",
                value="select * from %{schema} where userInput(@query) or ({label:\"title_label\", targetHits:100}nearestNeighbor(title_embedding, embedding)) or ({label:\"chunks_label\", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))"
            ),
            QueryTypeField(
                name="hits",
                type="integer",
                value="10"
            ),
            QueryTypeField(
                name="ranking.profile",
                type="string",
                value="learned-linear"
            ),
            QueryTypeField(
                name="presentation.summary",
                type="string",
                value="top_3_chunks"
            )
        ]
    )
    
    # Create the query profile
    query_profile = QueryProfile(
        id="hybrid",
        type=query_profile_type
    )
    
    return query_profile