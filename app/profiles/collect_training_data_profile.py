from vespa.package import (
    RankProfile,
    FirstPhaseRanking,
    SecondPhaseRanking,
)

from .base_features_profile import create_base_features_profile

def create_collect_training_data_profile():
    profile = RankProfile(
        name = "collect-training-data",
        inherits = "base_features",  # Inherit by name, not by object
        match_features=[
            "bm25(title)",
            "bm25(chunks)",
            "max_chunk_sim_scores",
            "max_chunk_text_scores",
            "avg_top_3_chunk_sim_scores",
            "avg_top_3_chunk_text_scores"
        ],
        first_phase=FirstPhaseRanking(
            expression=(
                "bm25(title) + "
                "bm25(chunks) + "
                "max_chunk_sim_scores() + "
                "max_chunk_text_scores()"
            ),
        ),
        second_phase=SecondPhaseRanking(
            expression="random"
        ),
    )
    return profile