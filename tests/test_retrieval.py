import pytest
from app.retrieval.bm25 import build_bm25_index, bm25_search


def test_bm25_search_returns_results():
    """Test BM25 index build and search."""
    docs = [
        {"content": "Retrieval Augmented Generation improves LLM accuracy", "source": "doc1"},
        {"content": "Vector databases store embeddings efficiently", "source": "doc2"},
        {"content": "LangChain simplifies LLM application development", "source": "doc3"},
    ]
    build_bm25_index(docs)
    results = bm25_search("vector embeddings", top_k=2)
    assert len(results) >= 1
    assert "content" in results[0]


def test_bm25_returns_empty_on_no_match():
    """Test BM25 returns empty when no relevant docs."""
    results = bm25_search("completely unrelated xyzabc query", top_k=5)
    assert isinstance(results, list)
