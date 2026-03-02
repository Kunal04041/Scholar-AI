from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


async def evaluate_response(query: str, answer: str, contexts: list) -> dict:
    """Run RAGAS evaluation on a query-answer-context triple."""
    try:
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
    except Exception as e:
        logger.error(f"[RAGAS] Evaluation failed: {e}")
        return {"error": str(e)}
