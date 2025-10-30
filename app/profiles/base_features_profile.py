from vespa.package import (
    RankProfile,
    Function,
)

def create_base_features_profile():
    profile  = RankProfile(
        name = "base_features",
        inputs = [("query(embedding)", "tensor<int8>(x[96])"), ("query(float_embedding)", "tensor<float>(x[768])")],
        rank_chunks = {
            "element-gap": 0
        }, 
        functions=[
            # Function to calculate BM25 scores for each chunk
            Function(
                name="chunk_text_scores",
                expression="elementwise(bm25(chunks),chunk,float)"
            ),
            
            # Function to get chunk embeddings (no unpacking needed for float tensors)
            Function(
                name="chunk_emb_vecs",
                expression="attribute(chunk_embeddings)"
            ),
            
            # Function to calculate dot product between query and chunk embeddings
            Function(
                name="chunk_dot_prod",
                expression="reduce(query(float_embedding) * chunk_emb_vecs(), sum, x)"
            ),
            
            # Function to calculate vector norms
            Function(
                name="vector_norms",
                expression="sqrt(sum(pow(t, 2), x))"
            ),
            
            # Function to calculate cosine similarity scores for chunks
            Function(
                name="chunk_sim_scores",
                expression="chunk_dot_prod() / (vector_norms(chunk_emb_vecs()) * vector_norms(query(float_embedding)))"
            ),
            
            # Function to get top 3 chunks by text scores
            Function(
                name="top_3_chunk_text_scores",
                expression="top(3, chunk_text_scores())"
            ),
            
            # Function to get top 3 chunks by similarity scores
            Function(
                name="top_3_chunk_sim_scores",
                expression="top(3, chunk_sim_scores())"
            ),
            
            # Function to calculate average of top 3 text scores
            Function(
                name="avg_top_3_chunk_text_scores",
                expression="reduce(top_3_chunk_text_scores(), avg, chunk)"
            ),
            
            # Function to calculate average of top 3 similarity scores
            Function(
                name="avg_top_3_chunk_sim_scores",
                expression="reduce(top_3_chunk_sim_scores(), avg, chunk)"
            ),
            
            # Function to get maximum text score across all chunks
            Function(
                name="max_chunk_text_scores",
                expression="reduce(chunk_text_scores(), max, chunk)"
            ),
            
            # Function to get maximum similarity score across all chunks
            Function(
                name="max_chunk_sim_scores",
                expression="reduce(chunk_sim_scores(), max, chunk)"
            )
        ],

    )
    return profile
