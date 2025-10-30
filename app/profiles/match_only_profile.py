from vespa.package import (
    RankProfile,
)

def create_match_only_profile():
    profile = RankProfile(
        name = "match-only",
        inputs = [("query(embedding)", "tensor<int8>(x[96])")],
    )
    return profile