from app.retrieval.hybrid import hybrid_search
import logging

logger = logging.getLogger(__name__)


async def retriever_agent(query: str, top_k: int = 5) -> list:
    """Retriever agent — runs hybrid BM25 + vector search."""
    logger.info(f"[Retriever] Searching for: {query}")
    results = await hybrid_search(query=query, top_k=top_k)
    logger.info(f"[Retriever] Found {len(results)} docs")
    return results
