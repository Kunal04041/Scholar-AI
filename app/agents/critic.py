from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research critic and synthesizer agent.
    Given a user query and retrieved document chunks, produce a comprehensive,
    well-cited answer. For each claim, reference the source like [Source: {{source}}].
    Be factual, concise, and structured. If the documents don't contain enough
    information, clearly state that."""),
    ("human", """Query: {query}

Retrieved Documents:
{context}

Provide a comprehensive answer with inline citations.""")
])


def _extract_text(response) -> str:
    """Safely extract text content from an LLM response."""
    content = response.content
    # New Gemini models return a list of blocks: [{'type': 'text', 'text': '...'}]
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    # Standard case: content is already a string
    return str(content)


async def critic_agent(query: str, docs: list, llm, stream: bool = False):
    """Critic agent — synthesizes final answer from retrieved docs."""
    logger.info(f"[Critic] Synthesizing answer from {len(docs)} docs")

    context = "\n\n".join([
        f"[Doc {i+1} | Source: {doc.get('source', 'unknown')}]\n{doc['content']}"
        for i, doc in enumerate(docs)
    ])

    critic_chain = CRITIC_PROMPT | llm

    if stream:
        async def stream_generator():
            async for chunk in critic_chain.astream({"query": query, "context": context}):
                yield {"chunk": _extract_text(chunk), "done": False}
            yield {"chunk": "", "done": True}
        return stream_generator()

    response = await critic_chain.ainvoke({"query": query, "context": context})

    return {
        "answer": _extract_text(response),
        "sources": [
            {"content": d["content"], "source": d.get("source", "unknown"), "score": d.get("score", 0.0)}
            for d in docs
        ],
    }
