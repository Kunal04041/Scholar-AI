from langchain_core.prompts import ChatPromptTemplate
from app.core.llm import get_llm
from app.agents.retriever import retriever_agent
from app.agents.critic import critic_agent
from app.evaluation.ragas_eval import evaluate_response
import logging

logger = logging.getLogger(__name__)

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research planning agent. Given a user query, break it into
    clear sub-questions that will help retrieve comprehensive information.
    Return a JSON list of sub-questions. Keep it to 2-3 sub-questions max."""),
    ("human", "Query: {query}\n\nReturn ONLY a JSON array of sub-questions.")
])


async def run_pipeline(
    query: str,
    session_id: str,
    top_k: int = 5,
    evaluate: bool = False,
    stream: bool = False,
):
    """Main multi-agent orchestration pipeline."""
    llm = get_llm()
    logger.info(f"[Planner] Processing query: {query}")

    planner_chain = PLANNER_PROMPT | llm
    plan_response = await planner_chain.ainvoke({"query": query})

    sub_questions = [query]
    try:
        import json
        sub_questions = json.loads(plan_response.content)
        if not isinstance(sub_questions, list):
            sub_questions = [query]
    except Exception:
        pass

    logger.info(f"[Planner] Sub-questions: {sub_questions}")

    # Step 2: Retriever agent fetches relevant docs for each sub-question
    all_docs = []
    for sub_q in sub_questions:
        docs = await retriever_agent(sub_q, top_k=top_k)
        all_docs.extend(docs)

    # Deduplicate docs by content
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc["content"] not in seen:
            seen.add(doc["content"])
            unique_docs.append(doc)

    # Step 3: Critic agent synthesizes final answer
    result = await critic_agent(query=query, docs=unique_docs, llm=llm, stream=stream)

    # Step 4: Optional RAGAS evaluation
    if evaluate and not stream:
        eval_scores = await evaluate_response(
            query=query,
            answer=result["answer"],
            contexts=[d["content"] for d in unique_docs],
        )
        result["evaluation"] = eval_scores

    return result
