from app.retrieval.vector_store import vector_search
from app.retrieval.bm25 import bm25_search
from typing import List
import logging

logger = logging.getLogger(__name__)


async def hybrid_search(query: str, top_k: int = 5) -> List[dict]:
    """
    Reciprocal Rank Fusion (RRF) hybrid search:
    Combines BM25 keyword results + Qdrant vector results.
    """
    vector_results = await vector_search(query, top_k=top_k)
    bm25_results = bm25_search(query, top_k=top_k)

    # RRF scoring
    k = 60  # RRF constant
    scores = {}

    for rank, doc in enumerate(vector_results):
        key = doc["content"][:100]  # use content prefix as key
        scores[key] = scores.get(key, {"doc": doc, "score": 0})
        scores[key]["score"] += 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_results):
        key = doc["content"][:100]
        scores[key] = scores.get(key, {"doc": doc, "score": 0})
        scores[key]["score"] += 1 / (k + rank + 1)

    # Sort by RRF score
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    results = [{**item["doc"], "score": item["score"]} for item in ranked[:top_k]]

    logger.info(f"[Hybrid] Returning {len(results)} fused results")
    return results
