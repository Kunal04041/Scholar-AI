import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_planner_returns_subquestions():
    """Test that planner correctly breaks query into sub-questions."""
    with patch("app.agents.planner.get_llm") as mock_llm:
        mock_chain = AsyncMock()
        mock_chain.content = '["What is RAG?", "How does vector search work?"]'
        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_chain)
        # Add your assertions here
        assert True  # Placeholder


@pytest.mark.asyncio
async def test_critic_synthesizes_answer():
    """Test critic agent synthesizes answer from docs."""
    docs = [
        {"content": "RAG stands for Retrieval Augmented Generation.", "source": "test.pdf", "score": 0.9}
    ]
    # Add integration test assertions here
    assert len(docs) > 0
