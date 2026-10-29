from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import requests
import api

from fastapi.testclient import TestClient

client = TestClient(api.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_documents():
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()
    assert "documents" in data
    assert data["count"] > 0
    assert len(data["documents"]) == data["count"]


def test_search():
    response = client.post(
        "/search",
        json={
            "query": "thermal exposure",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "thermal exposure"
    assert "results" in data
    assert data["result_count"] == len(data["results"])


def test_search_rejects_invalid_top_k():
    response = client.post(
        "/search",
        json={
            "query": "thermal exposure",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_ask(monkeypatch):
    """正常提问时，应返回答案和来源。"""

    def fake_answer_question(query, retriever, top_k=3):
        return {
            "query": query,
            "answer": "Thermal exposure affects urban heat vulnerability.",
            "sources": [
                {
                    "source": "sample.md",
                    "chunk_id": 0,
                    "text": "Thermal exposure affects urban heat vulnerability.",
                }
            ],
        }

    monkeypatch.setattr(api, "answer_question", fake_answer_question)

    response = client.post(
        "/ask",
        json={
            "query": "What affects urban heat exposure?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "What affects urban heat exposure?"
    assert data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "sample.md"


def test_ask_rejects_invalid_top_k():
    """top_k 超出允许范围时，应由 Pydantic 拒绝请求。"""

    response = client.post(
        "/ask",
        json={
            "query": "What affects urban heat exposure?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_ask_blank_query():
    """只有空格的问题应被业务逻辑拒绝。"""

    response = client.post(
        "/ask",
        json={
            "query": "   ",
            "top_k": 3,
        },
    )

    assert response.status_code == 400


def test_ask_handles_ollama_request_error(monkeypatch):
    """Ollama 请求失败时，API 应返回 502。"""

    def fake_answer_question(query, retriever, top_k=3):
        raise requests.Timeout("Ollama request timed out")

    monkeypatch.setattr(api, "answer_question", fake_answer_question)

    response = client.post(
        "/ask",
        json={
            "query": "What affects urban heat exposure?",
            "top_k": 3,
        },
    )

    assert response.status_code == 502
    assert "Ollama" in response.json()["detail"]
