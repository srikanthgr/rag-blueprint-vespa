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
            ]
        ),

        # Fields outside document block (for computed/embedded fields)
        fields = [
            # Vector embedding for document title (using 768 dimensions to match query)
            # Must be outside document block because it uses 'embed' in indexing
            Field(
                name = "title_embedding",
                type = "tensor<float>(x[768])",
                indexing = "input title | embed | attribute | index",
                ann = HNSW(distance_metric = "angular"),
                attribute = ["fast-search"]
            ),

            # Array of text chunks
            # Must be outside document block because it uses 'chunk' in indexing
            Field(
                name = "chunks",
                type = "array<string>",
                indexing = "input text | chunk fixed-length 1024 | summary | index",
                index = "enable-bm25"
            ),

            # Vector embeddings for each chunk (using 768 dimensions to match query)
            # Must be outside document block because it uses 'embed' in indexing
            Field(
                name = "chunk_embeddings",
                type = "tensor<float>(chunk{}, x[768])",
                indexing = "input text | chunk fixed-length 1024 | embed | attribute | index",
                ann = HNSW(distance_metric = "angular"),
                attribute = ["fast-search"]
            ),
        ],
        
        # Fieldset defines which fields are searched by default
        fieldsets = [
            FieldSet(
                name = "default",
                fields = [
                    "title",  # Only document fields can be in fieldsets
                ],
            ),
        ], 
    )
    
    return schema
