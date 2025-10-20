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
    SecondPhaseRanking,
)
import os

from collect_second_phase_profile import create_collect_second_phase_profile

# Get the directory where this file is located
MODELS_DIR = os.path.dirname(__file__)

# Define the path to the LightGBM model
LIGHTGBM_MODEL_PATH = os.path.join(MODELS_DIR, "lightgbm_model.json")

def create_second_with_gbdt_profile():
    profile = RankProfile(      
        name = "second-with-gbdt",
        inherits = create_collect_second_phase_profile(),
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
            expression = "lightgbm(" + LIGHTGBM_MODEL_PATH + ")",
        ),
    )
    return profile