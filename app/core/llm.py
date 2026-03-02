from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


def get_llm(use_fallback: bool = False):
    """Returns Gemini Flash as primary, Groq Llama as fallback."""
    if use_fallback or not settings.GEMINI_API_KEY:
        logger.info("Using Groq fallback LLM")
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="llama-3.1-70b-versatile",
            temperature=0.2,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
        streaming=True,
    )


def get_embeddings():
    """Returns Gemini Embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GEMINI_API_KEY,
    )
