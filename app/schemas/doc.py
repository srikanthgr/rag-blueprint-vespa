from vespa.package import (
    Field,
    Schema,
    Document,
    HNSW,
    FieldSet,
    DocumentSummary,
    Summary,
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
                    type = "bool",  # Vespa uses 'bool' not 'boolean'
                    indexing = "summary | attribute",
                ), 
                
                # Vector embedding for document title (using 768 dimensions to match query)
                Field(
                    name = "title_embedding",
                    type = "tensor<float>(x[768])",  # 768-dimensional to match float_embedding query
                    indexing = "input title | embed | attribute | index",
                    ann = HNSW(distance_metric = "angular")  # angular for float tensors
                ), 

                # Array of text chunks
                Field(
                    name = "chunks",
                    type="array<string>",
                    indexing = "input text | chunk fixed-length 1024 | summary | index",
                    index = "enable-bm25"
                ), 

                # Vector embeddings for each chunk (using 768 dimensions to match query)
                Field(
                    name = "chunk_embeddings",
                    type = "tensor<float>(chunk{}, x[768])",  # 768-dimensional to match float_embedding query
                    indexing = "input text | chunk fixed-length 1024 | embed | attribute | index",
                    ann = HNSW(distance_metric = "angular")
                ), 
            ]
        ),
        
        # Fieldset defines which fields are searched by default
        fieldsets = [
            FieldSet(
                name = "default",
                fields = [
                    "title",
                    "chunks",  # Fixed: was "chunk", should be "chunks"
                ],
            ),
        ], 

        # Document summaries for different use cases
        document_summaries = [
            # Document summary for basic document info (no special processing)
            DocumentSummary(
                name = "no_chunks",
                summary_fields=[
                    Summary(name="id"),  # Include document ID in summary
                    Summary(name="title"),  # Include title
                    Summary(name="created_timestamp"),  # Include creation timestamp
                    Summary(name="modified_timestamp"),  # Include modification timestamp
                    Summary(name="last_opened_timestamp"),  # Include last opened timestamp
                    Summary(name="open_count"),  # Include open count
                    Summary(name="favorite"),  # Include favorite status
                    Summary(name="chunks"),  # Include all chunks
                ]
            ), 

            # Document summary for top 3 most similar chunks
            DocumentSummary(
                name = "top_3_chunks",
                from_disk = True,  # Load from disk when needed (lazy loading)
                summary_fields = [
                    Summary(
                        name="chunks_top3", 
                        fields=[("source", "chunks")]  # Source field to select from
                        # Note: select_elements_by would be configured in rank profile
                    ),
                ],
            ),
        ], 
    )
    
    return schema
