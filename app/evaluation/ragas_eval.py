import logging
import os
from app.core.config import settings

logger = logging.getLogger(__name__)


async def evaluate_response(query: str, answer: str, contexts: list) -> dict:
    """Run RAGAS evaluation on a query-answer-context triple using Gemini."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from datasets import Dataset

        gemini_api_key = settings.GEMINI_API_KEY

        # Use Gemini as RAGAS judge instead of OpenAI
        ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=gemini_api_key,
            temperature=0,
        ))
        ragas_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=gemini_api_key,
        ))

        # Set judge on each metric
        faithfulness.llm = ragas_llm
        faithfulness.embeddings = ragas_embeddings
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings

        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        scores = result.to_pandas().to_dict(orient="records")[0]
        logger.info(f"[RAGAS] Scores: {scores}")
        return scores
    except ImportError as e:
        logger.warning(f"[RAGAS] Missing package: {e}")
        return {"error": "RAGAS not available. Run: pip install ragas datasets"}
    except Exception as e:
        logger.error(f"[RAGAS] Evaluation failed: {e}")
        return {"error": str(e)}
