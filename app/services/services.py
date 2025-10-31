# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""
Vespa Services Configuration using pyvespa

This module provides the Python equivalent of services.xml configuration for the RAG application.
It uses pyvespa's configuration services to define container, content clusters, and components.
"""

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
    transformer_input_ids,
    max_tokens,
    prepend,
    query,
    provider,
)

application_name = "rag"

def create_services():
    """
    Create Vespa services configuration for RAG application.

    This is the Python (pyvespa) equivalent of services.xml configuration.

    This configuration includes:
    - Container cluster with document processing and API
    - OpenAI LLM client for generation (with optional Vespa Cloud Secret Store support)
    - Nomic ModernBERT embedder for vector embeddings (768 dimensions)
    - RAG search chain for retrieval-augmented generation
    - Content cluster with minimum redundancy of 2
    - Minimum required Vespa version: 8.519.55

    Note: To use secrets from Vespa Cloud Secret Store, uncomment the secrets
    configuration in the component config and add apiKeySecretName parameter.
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
                
                # Search configuration (simplified - RAG chain removed for basic deployment)
                search(),
                
                # Container nodes
                nodes(
                    node(hostalias="node1")
                ),
                
                id="default",
                version="1.0"
            ),
            
            # Content cluster configuration
            # See https://docs.vespa.ai/en/reference/services-content.html
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

            version="1.0",
            minimum_required_vespa_version="8.519.55"
        ),
    )
    
    return services_config