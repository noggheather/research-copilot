
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

session = requests.Session()
session.trust_env = False

st.set_page_config(
    page_title="Research Copilot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Copilot")
st.caption("A lightweight research document retrieval system")

# 显示后端状态
try:
    health = session.get(
        f"{API_URL}/health",
        timeout=3,
    )
    health.raise_for_status()
    st.success("Backend connected")
except requests.RequestException as exc:
    st.error(f"Backend connection failed: {exc}")

st.divider()

# =========================
# 原有功能：普通文档检索
# =========================

st.header("🔎 Document Search")

query = st.text_input(
    "Search your research documents",
    placeholder="e.g. thermal exposure",
)

col1, col2 = st.columns(2)

with col1:
    method = st.selectbox(
        "Retrieval method",
        ["keyword", "bm25"],
        index=1,
    )

with col2:
    top_k = st.number_input(
        "Top-K results",
        min_value=1,
        max_value=20,
        value=3,
        key="search_top_k",
    )

if st.button("Search", type="primary"):

    if not query.strip():
        st.warning("Please enter a search query.")

    else:
        try:
            response = session.post(
                f"{API_URL}/search",
                json={
                    "query": query,
                    "top_k": int(top_k),
                    "method": method,
                },
                timeout=10,
            )

            response.raise_for_status()
            data = response.json()
            results = data["results"]

            st.subheader(
                f"Search Results ({len(results)})"
            )

            if not results:
                st.info("No matching passages found.")

            for rank, result in enumerate(results, start=1):
                with st.container(border=True):
                    st.markdown(
                        f"### {rank}. {result['source']}"
                    )

                    st.caption(
                        f"Chunk ID: {result['chunk_id']} | "
                        f"Score: {result['score']:.4f}"
                    )

                    st.write(result["text"])

        except requests.RequestException as exc:
            st.error(f"Search failed: {exc}")
        except (ValueError, KeyError, TypeError):
            st.error("Unexpected API response format.")


# =========================
# NEW: RAG 问答
# =========================

st.divider()
st.header("💬 Research Q&A")
st.caption(
    "Ask a question and get an answer based on your research documents."
)

question = st.text_area(
    "Enter your question",
    placeholder="e.g. What affects urban heat exposure?",
    height=100,
    key="rag_question",
)

rag_top_k = st.number_input(
    "Number of passages used for answering",
    min_value=1,
    max_value=20,
    value=3,
    key="rag_top_k",
)

if st.button("Ask AI", type="primary", key="ask_button"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Searching documents and generating an answer..."
            ):
                response = session.post(
                    f"{API_URL}/ask",
                    json={
                        "query": question,
                        "top_k": int(rag_top_k),
                    },
                    timeout=180,
                )

                response.raise_for_status()
                data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            sources = data.get("sources", [])

            st.subheader(f"Sources ({len(sources)})")

            if not sources:
                st.info("No source passages were returned.")

            for rank, source in enumerate(sources, start=1):
                with st.container(border=True):
                    st.markdown(
                        f"**{rank}. {source['source']}**"
                    )

                    st.caption(
                        f"Chunk ID: {source['chunk_id']}"
                    )

                    st.write(source["text"])

        except requests.Timeout:
            st.error(
                "The request timed out. The local model may need "
                "more time to generate an answer. Please try again."
            )
        except requests.HTTPError as exc:
            status_code = (
                exc.response.status_code
                if exc.response is not None
                else "unknown"
            )
            st.error(
                f"RAG request failed (HTTP {status_code}). "
                "Please check the backend logs."
            )
        except requests.RequestException as exc:
            st.error(f"Could not connect to the backend: {exc}")
        except (ValueError, KeyError, TypeError):
            st.error("Unexpected RAG API response format.")
