from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from typing import List
import logging

logger = logging.getLogger(__name__)


def chunk_documents(docs: List[dict]) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for split in splits:
            chunks.append({"content": split, "source": doc.get("source", "unknown")})

    logger.info(f"Chunked {len(docs)} docs into {len(chunks)} chunks")
    return chunks
