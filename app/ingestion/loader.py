from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from typing import Optional, List
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


async def load_document(url: Optional[str] = None, text: Optional[str] = None, source: str = "unknown") -> List[dict]:
    docs = []
    if text:
        docs.append({"content": text, "source": source})

    elif url:
        if url.endswith(".pdf"):
            loader = PyPDFLoader(url)
            pages = loader.load()
            docs = [{"content": p.page_content, "source": url} for p in pages]
        else:
            loader = WebBaseLoader(url)
            pages = loader.load()
            docs = [{"content": p.page_content, "source": url} for p in pages]

    logger.info(f"Loaded {len(docs)} document(s) from source: {source}")
    return docs
