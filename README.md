# 🎓 Scholar-AI — Multi-Agent Research Assistant

> A production-grade, fully cloud-based AI research assistant powered by multi-agent orchestration, hybrid retrieval, and real-time evaluation.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![Gemini](https://img.shields.io/badge/Gemini-1.5Flash-purple)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-red)

---

## 🧠 Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│        Planner Agent            │  ← Breaks query into sub-tasks
└────────────┬────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌────────────┐  ┌─────────────┐
│ Retriever  │  │  Web Search │  ← Hybrid BM25 + Vector Search
│   Agent    │  │   Agent     │
└─────┬──────┘  └──────┬──────┘
      │                │
      └──────┬──────────┘
             ▼
    ┌─────────────────┐
    │  Critic Agent   │  ← Evaluates & re-ranks answers
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Streaming API   │  ← FastAPI SSE streaming
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Streamlit UI    │
    └─────────────────┘
```

---

## ✨ Features

- 🤖 **Multi-Agent Pipeline** — Planner, Retriever, Critic agents using LangGraph
- 🔍 **Hybrid Retrieval** — BM25 keyword + Qdrant vector search combined
- 📊 **RAGAS Evaluation** — Live faithfulness, relevancy & recall scores
- ⚡ **Streaming Responses** — FastAPI SSE with sub-2s time-to-first-token
- 🔗 **Inline Citations** — Every answer chunk cites its source document
- 🌐 **100% Cloud** — Gemini API + Qdrant Cloud + Render + Streamlit Cloud
- 🔄 **Dual LLM** — Gemini 1.5 Flash (primary) + Groq Llama 3.1 70B (fallback)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Google Gemini 1.5 Flash |
| LLM Fallback | Groq (Llama 3.1 70B) |
| Embeddings | Gemini Embedding API |
| Vector DB | Qdrant Cloud |
| Orchestration | LangGraph |
| Backend | FastAPI + Python 3.11 |
| Storage | Supabase |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| Deployment | Render + Streamlit Cloud |
| CI/CD | GitHub Actions |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Kunal04041/Scholar-AI.git
cd Scholar-AI
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Run Backend
```bash
uvicorn app.main:app --reload
```

### 4. Run Frontend
```bash
streamlit run frontend/app.py
```

---

## 📁 Project Structure

```
Scholar-AI/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── schemas.py       # Pydantic models
│   ├── agents/
│   │   ├── planner.py       # Planner agent
│   │   ├── retriever.py     # Retriever agent
│   │   └── critic.py        # Critic/evaluator agent
│   ├── core/
│   │   ├── config.py        # Settings & env vars
│   │   └── llm.py           # LLM client (Gemini + Groq)
│   ├── retrieval/
│   │   ├── vector_store.py  # Qdrant operations
│   │   ├── bm25.py          # BM25 keyword search
│   │   └── hybrid.py        # Hybrid retrieval fusion
│   ├── ingestion/
│   │   ├── loader.py        # PDF/URL document loader
│   │   └── chunker.py       # Smart text chunking
│   └── evaluation/
│       └── ragas_eval.py    # RAGAS evaluation pipeline
├── frontend/
│   └── app.py               # Streamlit UI
├── tests/
│   ├── test_agents.py
│   └── test_retrieval.py
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI
├── .env.example
├── requirements.txt
├── Dockerfile
└── render.yaml              # Render deployment config
```

---

## 📈 Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 🔄 Run `evaluate.py` to generate |
| Answer Relevancy | 🔄 Run `evaluate.py` to generate |
| Context Recall | 🔄 Run `evaluate.py` to generate |

---

## 🔑 Environment Variables

See `.env.example` for all required keys.

---

## 📄 License

MIT License — feel free to use and adapt.
