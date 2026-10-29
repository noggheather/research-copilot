
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from rag import answer_question


@pytest.fixture
def fake_retriever():
    """Create a fake retriever without loading real documents."""

    class FakeRetriever:
        def search(self, query, top_k=3):
            return [
                {
                    "source": "sample.md",
                    "chunk_id": 0,
                    "text": (
                        "Urban heat vulnerability depends on "
                        "thermal exposure and the built environment."
                    ),
                    "score": 1.5,
                }
            ]

    return FakeRetriever()


def test_empty_query_raises_error(fake_retriever):
    """An empty query should be rejected."""

    with pytest.raises(ValueError):
        answer_question("   ", fake_retriever)


def test_no_results_skips_generation(monkeypatch):
    """The model should not be called when retrieval returns no results."""

    class EmptyRetriever:
        def search(self, query, top_k=3):
            return []

    def fake_generate_answer(query, context):
        raise AssertionError("Generation should not be called")

    monkeypatch.setattr(
        "rag.generate_answer",
        fake_generate_answer,
    )

    result = answer_question(
        "quantum spaceship",
        EmptyRetriever(),
    )

    assert result["answer"] == (
        "No relevant information was found in the documents."
    )
    assert result["sources"] == []


def test_successful_rag_returns_answer_and_sources(
    monkeypatch,
    fake_retriever,
):
    """Retrieved context should be passed to generation."""

    captured = {}

    def fake_generate_answer(query, context):
        captured["query"] = query
        captured["context"] = context
        return "Urban heat depends on thermal exposure."

    monkeypatch.setattr(
        "rag.generate_answer",
        fake_generate_answer,
    )

    result = answer_question(
        "What affects urban heat exposure?",
        fake_retriever,
        top_k=2,
    )

    assert result["query"] == "What affects urban heat exposure?"
    assert result["answer"] == (
        "Urban heat depends on thermal exposure."
    )

    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "sample.md"
    assert result["sources"][0]["chunk_id"] == 0

    assert captured["query"] == "What affects urban heat exposure?"
    assert "thermal exposure" in captured["context"]
    assert "sample.md" in captured["context"]
