# Query Function Examples

## Example Query: "data science with python"

### 1. Text-Only (`match_weakand_query_fn`)
**What it finds:**
- Documents with keywords: "data", "science", "python"
- Exact matches preferred

**Example Matches:**
- ✅ "Introduction to Data Science using Python" (exact keywords)
- ✅ "Python for Data Science course" (keyword match)
- ❌ "Machine learning tutorial" (no keywords, even if semantically related)

---

### 2. Semantic-Only (`match_semantic_query_fn`)
**What it finds:**
- Semantically similar documents
- May not contain exact keywords

**Example Matches:**
- ✅ "Introduction to Data Science using Python" (semantically similar)
- ✅ "Machine learning tutorial" (similar concept, different words)
- ✅ "Analytics with NumPy and Pandas" (related topic)
- ❌ "Cooking recipes" (not semantically related)

---

### 3. Hybrid (`match_hybrid_query_fn`)
**What it finds:**
- Documents matching keywords OR semantic similarity
- Best coverage

**Example Matches:**
- ✅ "Introduction to Data Science using Python" (both keyword + semantic)
- ✅ "Python for Data Science course" (keyword match)
- ✅ "Machine learning tutorial" (semantic match)
- ✅ "Analytics with NumPy and Pandas" (semantic match)
- ❌ "Cooking recipes" (neither keyword nor semantic match)

---

## Important Note: `ranking: "match-only"`

All three functions use `ranking: "match-only"`, which means:
- **They only do MATCHING, not RANKING**
- Results are returned in random/arbitrary order
- Used for evaluation/testing when you want pure matching without ranking bias
- For production, you'd use proper ranking profiles like `"bm25"`, `"semantic"`, or `"fusion"`

## Production Usage

To use these for actual search, change the ranking profile:

```python
# Text search with BM25 ranking
{
    "ranking": "bm25",  # Instead of "match-only"
    ...
}

# Semantic search with cosine similarity ranking
{
    "ranking": "semantic",  # Instead of "match-only"
    ...
}

# Hybrid search with fusion ranking
{
    "ranking": "fusion",  # Instead of "match-only"
    ...
}
```


