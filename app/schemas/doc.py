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
    DocumentSummary,
    SummaryFeature,
)

def create_docs_schema():
    schema = Schema(
        name = "doc",  # Schema name - defines the document type in Vespa
        document = Document(
            fields = [
                # Basic document identifier field
                Field(
                    name = "id",
                    type = "string",
                    indexing = "summary | attribute",  # summary: can be returned in search results, attribute: can be used for filtering/sorting
                ),

                # Document title field with full-text search capabilities
                Field(
                    name = "title",
                    type = "string",
                    indexing = "index | summary",  # index: enables full-text search, summary: can be returned in results
                    index = "enable-bm25",  # Enables BM25 ranking algorithm for this field
                ),

                # Main content field (no indexing specified - just stored)
                Field(
                    name = "text",
                    type = "string",
                ), 

                # Timestamp when document was created
                Field(
                    name = "created_timestamp",
                    type = "long",
                    indexing = "summary | attribute",  # Can be returned and used for filtering/sorting
                ),

                # Timestamp when document was last modified
                Field(
                    name = "modified_timestamp",
                    type = "long",
                    indexing = "summary | attribute",
                ), 

                # Timestamp when document was last opened by user
                Field(
                    name = "last_opened_timestamp",
                    type = "long",
                    indexing = "summary | attribute",
                ), 

                # Counter for how many times document was opened
                Field(
                    name = "open_count",
                    type = "int",
                    indexing = "summary | attribute",
                ), 

                # Boolean flag indicating if document is favorited
                Field(
                    name = "favorite",
                    type = "boolean",
                    indexing = "summary | attribute",
                ), 

                # Vector embedding for document title (96-dimensional int8 tensor)
                Field(
                    name = "title_embedding",
                    type = "tensor<int8>(x[96])",  # 96-dimensional vector with int8 precision
                    indexing = "input title | embed | packbits | attribute | index",  # embed: generates embedding from title, packbits: compresses for storage
                    ann = HNSW(  # Approximate Nearest Neighbor search using Hierarchical Navigable Small World
                        distance_metric = "hamming",  # Uses Hamming distance for similarity (good for int8 vectors)
                    )
                ), 

                # Array of text chunks created from the main text field
                Field(
                    name = "chunks",
                    type="array<string>",  # Array of string chunks
                    indexing = "input text | chunk fixed-length 1024 | summary | index",  # chunk: splits text into 1024-char chunks
                    index = "enable-bm25",  # Enables BM25 search on each chunk
                ), 

                # Vector embeddings for each chunk (tensor with chunk dimension and 96-dimensional vectors)
                Field(
                    name = "chunks_embedding",  # Note: should be "chunk_embeddings" to match original schema
                    type = "tensor<int8>(chunk{}, x[96])",  # Tensor with chunk dimension and 96-dim vectors
                    indexing = "input text | chunk fixed-length 1024 | embed | pack_bits | attribute | index",  # Creates embeddings for each chunk
                    ann = HNSW(  # ANN search for chunk embeddings
                        distance_metric = "hamming",
                    ),                     
                ), 
            ], 

            # Fieldset defines which fields are searched by default
            fieldsets = [
                FieldSet(
                    name = "default",
                    fields = [
                        "title",
                        "chunk",  # Note: should be "chunks" to match field name
                    ],
                ),
            ], 

            # Document summary for basic document info (no special processing)
            document_summary = DocumentSummary(
                name = "no_chunks",
                fields=[
                    SummaryFeature(name="id"),  # Include document ID in summary
                    SummaryFeature(name="title"),  # Include title
                    SummaryFeature(name="created_timestamp"),  # Include creation timestamp
                    SummaryFeature(name="modified_timestamp"),  # Include modification timestamp
                    SummaryFeature(name="last_opened_timestamp"),  # Include last opened timestamp
                    SummaryFeature(name="open_count"),  # Include open count
                    SummaryFeature(name="favorite"),  # Include favorite status
                    SummaryFeature(name="chunks"),  # Include all chunks
                ]
            ), 

            # Document summary for top 3 most similar chunks
            document_summary = DocumentSummary(
                name = "top_3_chunks",
                from_disk = True,  # Load from disk when needed (lazy loading)
                fields = [
                    SummaryFeature(
                        name="chunks_top3", 
                        source="chunks",  # Source field to select from
                        select_elements_by="top_3_chunk_sim_scores"  # Function to select top 3 chunks by similarity
                    ),
                ],
            ), 
        ), 
    )

