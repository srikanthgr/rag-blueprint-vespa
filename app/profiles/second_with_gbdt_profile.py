from vespa.package import (
    RankProfile,
    SecondPhaseRanking,
)
import os

from .collect_second_phase_profile import create_collect_second_phase_profile

def create_second_with_gbdt_profile():
    # Note: The lightgbm model file path is relative to the models directory in the application package
    # The model file should be named "lightgbm_model.json" and placed in the models directory
    profile = RankProfile(      
        name = "second-with-gbdt",
        inherits = "collect-second-phase",  # Inherit by name, not by object
        match_features = [
            "max_chunk_sim_scores",
            "max_chunk_text_scores",
            "avg_top_3_chunk_text_scores",
            "avg_top_3_chunk_sim_scores",
            "bm25(title)",
            "modified_freshness",
            "open_count",
            "firstPhase"
        ],
        rank_features = [
            "nativeProximity",
            "nativeFieldMatch",
            "nativeRank",
            "elementSimilarity(chunks)",
            "fieldTermMatch(chunks, 4).firstPosition",
            "fieldTermMatch(chunks, 4).occurrences",
            "fieldTermMatch(chunks, 4).weight",
            "term(3).significance"
        ],
        second_phase = SecondPhaseRanking(
            expression = "lightgbm('lightgbm_model.json')",  # Model name only, not full path
        ),
    )
    return profile