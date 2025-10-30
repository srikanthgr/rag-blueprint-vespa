from vespa.package import (
    QueryProfile,
    QueryProfileType,
    QueryTypeField,
)
from hybrid import create_hybrid_query_profile as create_hybrid_query_profile


def create_hybrid_with_gbdt_query_profile():
    # Create the query profile type with GBDT-specific fields
    query_profile_type = QueryProfileType(
        fields=[
            QueryTypeField(
                name="hits",
                type="integer",
                value="20"  # Override hits from hybrid profile (was 10)
            ),
            QueryTypeField(
                name="ranking.profile",
                type="string",
                value="second-with-gbdt"  # Use GBDT ranking profile
            ),
            QueryTypeField(
                name="presentation.summary",
                type="string",
                value="top_3_chunks"  # Return top 3 chunks summary
            )
        ]
    )
    
    # Create the query profile that inherits from hybrid
    query_profile = QueryProfile(
        id="hybrid-with-gbdt",
        inherits=create_hybrid_query_profile(),  # Inherit from the hybrid query profile
        type=query_profile_type
    )
    
    return query_profile