from vespa.package import ServicesConfiguration
from vespa.configuration.services import (
    services,
    container,
    component,
    config,
    search,
    chain,
    searcher,
    document_api,
    document_processing,
    content,
    min_redundancy,
    documents,
    document,
    node,
    nodes,
    transformer_model,
    tokenizer_model,
    transformer_output,
    max_tokens,
    prepend,
    query,
    provider,
)

application_name = "rag"

def create_services():
    """
    Create Vespa services configuration for RAG application.
    
    This configuration includes:
    - OpenAI LLM client for generation
    - Nomic ModernBERT embedder for vector embeddings
    - RAG search chain for retrieval-augmented generation
    - Document processing and API
    - Content cluster with redundancy
    """
    
    services_config = ServicesConfiguration(
        application_name=application_name,
        services_config= services(
            # Container cluster configuration
            container(
                # Document processing pipeline
                document_processing(),
                
                # Document API for feeding documents
                document_api(),
                
                # OpenAI component for LLM integration
                component(
                    config(
                        name="ai.vespa.llm.clients.llm-client"
                        # Uncomment to use secret from Vespa Cloud Secret Store
                        # apiKeySecretName="openai-api-key"
                    ),
                    id="openai",
                    class_="ai.vespa.llm.clients.OpenAI"
                ),
                
                # Nomic ModernBERT embedder component
                component(
                    transformer_model(
                        url="https://data.vespa-cloud.com/onnx_models/nomic-ai-modernbert-embed-base/model.onnx"
                    ),
                    tokenizer_model(
                        url="https://data.vespa-cloud.com/onnx_models/nomic-ai-modernbert-embed-base/tokenizer.json"
                    ),
                    transformer_output("token_embeddings"),
                    max_tokens("8192"),
                    prepend(
                        query("search_query:"),
                        document("search_document:")
                    ),
                    id="nomicmb",
                    type_="hugging-face-embedder"
                ),
                
                # Search configuration with RAG chain
                search(
                    chain(
                        searcher(
                            config(
                                provider(id="openai"),
                                name="ai.vespa.search.llm.llm-searcher"
                            ),
                            id="ai.vespa.search.llm.RAGSearcher"
                        ),
                        id="openai",
                        inherits="vespa"
                    )
                ),
                
                # Container nodes
                nodes(
                    node(hostalias="node1")
                ),
                
                id="default",
                version="1.0"
            ),
            
            # Content cluster configuration
            content(
                # Minimum redundancy for high availability
                min_redundancy("2"),
                
                # Document configuration
                documents(
                    document(
                        type_="doc",
                        mode="index"
                    )
                ),
                # Content nodes
                nodes(
                    node(
                        hostalias="node1",
                        distribution_key="0"
                    )
                ),
                
                id="content",
                version="1.0"
            ),
            
            version="1.0"
        ),
    )
    
    return services_config