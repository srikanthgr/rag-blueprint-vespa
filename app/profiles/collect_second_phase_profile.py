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
    FirstPhaseRanking,
)

from collect_training_data_profile import create_collect_training_data_profile

def create_collect_second_phase_profile():
    profile = RankProfile(
        name="collect-second-phase",
        inherits=create_collect_training_data_profile(),
        match_features=[
            "bm25(title)",
            "bm25(chunks)",
            "max_chunk_sim_scores",
            "max_chunk_text_scores",
            "avg_top_3_chunk_sim_scores",    
            "avg_top_3_chunk_text_scores",
            "modified_freshness",
            "is_favorite",
            "open_count"
        ],
        rank_properties={
            "freshness(modified_timestamp).maxAge": 94672800,  # 3 years in seconds
        },
        functions=[
            Function(
                name="modified_freshness",
                expression="freshness(modified_timestamp)"
            ),
            Function(
                name="is_favorite",
                expression="if(attribute(favorite), 1.0, 0.0)"
            ),
            Function(
                name="open_count",
                expression="attribute(open_count)"
            ),
        ],
        first_phase=FirstPhaseRanking(
            expression=(
                "-7.798639+13.383840*avg_top_3_chunk_sim_scores+0.203145*avg_top_3_chunk_text_scores+0.159914*bm25(chunks)+0.191867*bm25(title)+10.067169*max_chunk_sim_scores+0.153392*max_chunk_text_scores"
            )
        ),
    )
    return profile