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
        name = "doc", 
        document = Document(
            fields = [
                Field(
                    name = "id",
                    type = "string",
                    indexing = "summary | attribute",
                ),

                Field(
                    name = "title",
                    type = "string",
                    indexing = "index | summary",
                    index = "enable-bm25",
                ),

                Field(
                    name = "text",
                    type = "string",
                ), 

                Field(
                    name = "created_timestamp",
                    type = "long",
                    indexing = "summary | attribute",
                ),

                Field(
                    name = "modified_timestamp",
                    type = "long",
                    indexing = "summary | attribute",
                ), 

                Field(
                    name = "last_opened_timestamp",
                    type = "long",
                    indexing = "summary | attribute",
                ), 

                Field(
                    name = "open_count",
                    type = "int",
                    indexing = "summary | attribute",
                ), 

                Field(
                    name = "favorite",
                    type = "boolean",
                    indexing = "summary | attribute",
                ), 

                Field(
                    name = "title_embedding",
                    type = "tensor<int8>(x[96])",
                    indexing = "input title | embed | packbits | attribute | index",
                    ann = HNSW(
                        distance_metric = "hamming",
                    )
                ), 
                Field(
                    name = "chunks",
                    type="array<string>",
                    indexing = "input text | chunk fixed-length 1024 | summary | index",
                    index = "enable-bm25",
                )
            ]
        ), 
    )