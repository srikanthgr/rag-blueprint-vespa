# Vespa Field Expressions

## Overview
Vespa field expressions control how data is stored and accessed in your search engine. Each expression determines the storage and retrieval behavior of a field.

## Field Expression Types

| Expression | Description | Use Case | Example |
|------------|-------------|----------|---------|
| `attribute` | Writes the execution value to the current field. During deployment, this indicates that the field should be stored as an attribute. | Fast random access, filtering, sorting | User IDs, timestamps, categories |
| `index` | Writes the execution value to the current field. During deployment, this indicates that the field should be stored as an index field. | Full-text search, phrase matching | Document content, titles, descriptions |
| `summary` | Writes the execution value to the current field. During deployment, this indicates that the field should be included in the document summary. | Result display, metadata | Snippets, highlights, metadata |

## What does "stored as an attribute" mean?

When a field is stored as an **attribute** in Vespa:

1. **Fast Access**: The field is stored in a way that allows very fast random access
2. **Memory Storage**: Typically stored in memory for quick retrieval
3. **Filtering & Sorting**: Optimized for filtering, sorting, and grouping operations
4. **Exact Matching**: Best for exact value matches rather than text search

### Attribute Example:
```yaml
# In your schema definition
field user_id type string {
    indexing: attribute
}
field created_at type long {
    indexing: attribute
}
field category type string {
    indexing: attribute
}
```

**Use cases for attributes:**
- User IDs for filtering results by user
- Timestamps for date range filtering
- Categories for faceted search
- Numeric values for sorting
- Status flags for filtering

## Index vs Attribute vs Summary

### Index Field
- **Purpose**: Full-text search and phrase matching
- **Storage**: Inverted index for fast text search
- **Use**: Searchable content, titles, descriptions

```yaml
field title type string {
    indexing: index | summary
}
field content type string {
    indexing: index | summary
}
```

### Summary Field
- **Purpose**: Include in search result summaries
- **Storage**: Stored for retrieval in results
- **Use**: Display metadata, snippets, highlights

```yaml
field snippet type string {
    indexing: summary
}
field author type string {
    indexing: attribute | summary
}
```

## Complete Example Schema

```yaml
schema document {
    document document {
        # Attribute fields - fast access, filtering, sorting
        field user_id type string {
            indexing: attribute
        }
        field created_at type long {
            indexing: attribute
        }
        field category type string {
            indexing: attribute
        }
        field priority type int {
            indexing: attribute
        }
        
        # Index fields - full-text searchable
        field title type string {
            indexing: index | summary
        }
        field content type string {
            indexing: index | summary
        }
        
        # Summary fields - included in results
        field snippet type string {
            indexing: summary
        }
        field author type string {
            indexing: attribute | summary
        }
    }
}
```

## Performance Considerations

- **Attributes**: Fast for filtering/sorting, but consume more memory
- **Indexes**: Fast for text search, but slower for exact matches
- **Summary**: Required for fields you want to return in search results
- **Combined**: You can combine expressions like `attribute | summary` for fields that need both fast access and result display 


## Attribute Explanations

### Field Attributes:
- **`name`**: Field identifier - unique name for the field in your schema
- **`type`**: Data type (string, long, int, boolean, tensor, array) - defines what kind of data the field stores
- **`indexing`**: How the field is processed and stored - controls storage behavior and search capabilities
  - `summary`: Can be returned in search results - fields included in document summaries
  - `attribute`: Can be used for filtering, sorting, grouping - fast random access, stored in memory
  - `index`: Enables full-text search - creates inverted index for text search
  - `input`: Field accepts input data - allows data to be written to this field
  - `embed`: Generates vector embeddings - creates dense vector representations
  - `packbits`/`pack_bits`: Compresses vector data for storage - reduces memory usage for vectors
  - `chunk`: Splits text into fixed-length chunks - breaks large text into smaller searchable pieces

### Tensor Types:
- **`tensor<int8>(x[96])`**: 96-dimensional vector with 8-bit integers - compact vector representation
- **`tensor<int8>(chunk{}, x[96])`**: Tensor with chunk dimension and 96-dim vectors - multiple vectors per document

### HNSW (Approximate Nearest Neighbor):
- **`distance_metric`**: Similarity measure (hamming for int8, euclidean for float) - how vector similarity is calculated
- Enables fast vector similarity search - allows finding similar documents quickly

### Document Summaries:
- **`from_disk`**: Lazy loading from disk storage - loads data only when needed
- **`select_elements_by`**: Function to select specific elements from arrays - custom selection logic

### RAG System Design:
This schema is designed for a RAG (Retrieval-Augmented Generation) system where:
- Documents are chunked into smaller pieces for better retrieval
- Each chunk gets vector embeddings for semantic search
- Documents can be searched both by text (BM25) and semantic similarity (vector search)
- Supports hybrid search combining multiple retrieval methods