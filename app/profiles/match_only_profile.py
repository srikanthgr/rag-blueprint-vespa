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
)

def create_match_only_profile():
    profile = RankProfile(
        name = "match-only",
        inputs = [("query(embedding)", "tensor<int8>(x[96])")],
    )
    return profile