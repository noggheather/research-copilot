
import requests


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen3:1.7b"


def build_context(results: list[dict]) -> str:
    """Build a context string from retrieved document chunks."""
    context_parts = []

    for result in results:
        part = (
            f"Source: {result['source']}\n"
            f"Chunk ID: {result['chunk_id']}\n"
            f"Text: {result['text']}"
        )
        context_parts.append(part)

    return "\n\n".join(context_parts)


def generate_answer(query: str, context: str) -> str:
    """Generate an answer using the local Ollama model."""
    prompt = f"""
You are a research assistant.
Answer the question using only the provided context.
If the context does not contain enough information, say so.
Do not invent facts.

Context:
{context}

Question:
{query}

Answer in the same language as the question.
"""

    session = requests.Session()
    session.trust_env = False

    response = session.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )

    response.raise_for_status()
    data = response.json()

    return data["response"].strip()


def answer_question(
    query: str,
    retriever,
    top_k: int = 3,
) -> dict:
    """Retrieve relevant chunks and generate an answer."""
    if not query.strip():
        raise ValueError("query must not be empty")

    results = retriever.search(query, top_k=top_k)

    if not results:
        return {
            "query": query,
            "answer": "No relevant information was found in the documents.",
            "sources": [],
        }

    context = build_context(results)
    answer = generate_answer(query, context)

    sources = [
        {
            "source": result["source"],
            "chunk_id": result["chunk_id"],
            "text": result["text"],
        }
        for result in results
    ]

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
    }
