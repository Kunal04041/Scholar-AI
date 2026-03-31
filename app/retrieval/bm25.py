from rank_bm25 import BM25Okapi
from typing import List
import logging

logger = logging.getLogger(__name__)

_bm25_index = None
_bm25_corpus = []


def build_bm25_index(documents: List[dict]):
    """Build BM25 index from document content."""
    global _bm25_index, _bm25_corpus
    _bm25_corpus = documents
    tokenized = [doc["content"].lower().split() for doc in documents]
    _bm25_index = BM25Okapi(tokenized)
    logger.info(f"BM25 index built with {len(documents)} documents")


def bm25_search(query: str, top_k: int = 5) -> List[dict]:
    """BM25 keyword search over in-memory index."""
    if _bm25_index is None or not _bm25_corpus:
        logger.warning("BM25 index not built yet, returning empty results")
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {**_bm25_corpus[i], "score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0
    ]
