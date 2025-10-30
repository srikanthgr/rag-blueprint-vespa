from vespa.package import (
    RankProfile,
    FirstPhaseRanking,
)

from .base_features_profile import create_base_features_profile

def create_learned_linear_profile():
    profile = RankProfile(
        name="learned-linear",
        inherits="base_features",  # Inherit by name, not by object
        inputs=[
            ("query(embedding)", "tensor<int8>(x[96])"),
            ("query(float_embedding)", "tensor<float>(x[768])"),
            ("query(intercept)", "double"),
            ("query(avg_top_3_chunk_sim_scores_param)", "double"),
            ("query(avg_top_3_chunk_text_scores_param)", "double"),
            ("query(bm25_chunks_param)", "double"),
            ("query(bm25_title_param)", "double"),
            ("query(max_chunk_sim_scores_param)", "double"),
            ("query(max_chunk_text_scores_param)", "double")
        ],
        first_phase=FirstPhaseRanking(
            expression=(
                "query(intercept) + "
                "query(avg_top_3_chunk_sim_scores_param) * avg_top_3_chunk_sim_scores() + "
                "query(avg_top_3_chunk_text_scores_param) * avg_top_3_chunk_text_scores() + "
                "query(bm25_title_param) * bm25(title) + "
                "query(bm25_chunks_param) * bm25(chunks) + "
                "query(max_chunk_sim_scores_param) * max_chunk_sim_scores() + "
                "query(max_chunk_text_scores_param) * max_chunk_text_scores()"
            )
        ),
        summary_features=[
            "top_3_chunk_sim_scores"
        ]
    )
    return profile