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
        model="gemini-3-flash-preview",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
        streaming=True,
    )


def get_embeddings():
    """Returns Gemini Embedding model."""
    if not settings.GEMINI_API_KEY or "your_gemini_api_key_here" in settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set or is a placeholder.")
        raise ValueError("GEMINI_API_KEY is missing. Please set it in the .env file.")
        
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        google_api_key=settings.GEMINI_API_KEY,
    )
