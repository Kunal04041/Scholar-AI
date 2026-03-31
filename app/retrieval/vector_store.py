from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import settings
from app.core.llm import get_embeddings
import uuid
import logging

logger = logging.getLogger(__name__)


def get_qdrant_client():
    if not settings.QDRANT_URL or "your_qdrant_cloud_url_here" in settings.QDRANT_URL:
        logger.warning("QDRANT_URL is not set or is a placeholder. Using local in-memory instance.")
        return QdrantClient(":memory:")
    
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


async def ensure_collection():
    client = get_qdrant_client()
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    if settings.QDRANT_COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")


async def store_chunks(chunks: list) -> int:
    """Embed and store document chunks in Qdrant."""
    await ensure_collection()
    embeddings_model = get_embeddings()
    client = get_qdrant_client()

    texts = [c["content"] for c in chunks]
    vectors = embeddings_model.embed_documents(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"content": chunk["content"], "source": chunk.get("source", "unknown")},
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=points)
    logger.info(f"Stored {len(points)} chunks in Qdrant")
    return len(points)


async def vector_search(query: str, top_k: int = 5) -> list:
    """Search Qdrant for similar document chunks."""
    embeddings_model = get_embeddings()
    client = get_qdrant_client()

    query_vector = embeddings_model.embed_query(query)
    results = client.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {"content": r.payload["content"], "source": r.payload.get("source", "unknown"), "score": r.score}
        for r in results
    ]
