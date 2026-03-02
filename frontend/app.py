import streamlit as st
import requests
import json
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Scholar-AI",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Scholar-AI — Multi-Agent Research Assistant")
st.caption("Powered by Gemini 1.5 Flash + LangGraph + Qdrant Hybrid Search")

# Sidebar
with st.sidebar:
    st.header("📂 Ingest Documents")
    ingest_url = st.text_input("Document URL (PDF or webpage)")
    source_name = st.text_input("Source Name", value="my-doc")
    if st.button("Ingest"):
        with st.spinner("Ingesting..."):
            res = requests.post(f"{API_BASE}/ingest", json={"url": ingest_url, "source_name": source_name})
            if res.ok:
                data = res.json()
                st.success(f"✅ Stored {data['chunks_stored']} chunks from {data['source']}")
            else:
                st.error(f"❌ Error: {res.text}")

    st.divider()
    enable_eval = st.checkbox("📊 Enable RAGAS Evaluation", value=False)
    top_k = st.slider("Top-K Documents", min_value=3, max_value=10, value=5)

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            res = requests.post(
                f"{API_BASE}/query",
                json={"query": prompt, "top_k": top_k, "evaluate": enable_eval},
            )
            if res.ok:
                data = res.json()
                answer = data["answer"]
                st.markdown(answer)

                # Show sources
                if data.get("sources"):
                    with st.expander("📚 Sources"):
                        for src in data["sources"]:
                            st.markdown(f"**{src['source']}** (score: {src['score']:.3f})")
                            st.caption(src["content"][:300] + "...")

                # Show RAGAS scores
                if data.get("evaluation"):
                    with st.expander("📊 RAGAS Evaluation"):
                        st.json(data["evaluation"])

                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(f"❌ Error: {res.text}")
