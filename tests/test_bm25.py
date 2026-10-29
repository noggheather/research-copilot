from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import pytest
from bm25_retriever import BM25Retriever


def test_bm25_search():
    chunks = [
        {
            "chunk_uid": 0,
            "source": "heat.md",
            "chunk_id": 0,
            "text": "Urban thermal exposure and heat risk.",
        },
        {
            "chunk_uid": 1,
            "source": "computer.md",
            "chunk_id": 0,
            "text": "Python software engineering and computer systems.",
        },
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("thermal exposure")

    assert len(results) == 1
    assert results[0]["source"] == "heat.md"
    assert isinstance(results[0]["score"], float)


def test_bm25_empty_query():
    chunks = [
        {
            "chunk_uid": 0,
            "source": "heat.md",
            "chunk_id": 0,
            "text": "Urban thermal exposure.",
        }
    ]

    retriever = BM25Retriever(chunks)

    assert retriever.search("") == []


def test_bm25_invalid_top_k():
    retriever = BM25Retriever([])

    with pytest.raises(ValueError):
        retriever.search("heat", top_k=0)


def test_bm25_tokenization_preserves_repeated_words():
    chunks = [
        {
            "chunk_uid": 0,
            "source": "heat.md",
            "chunk_id": 0,
            "text": "heat heat heat exposure",
        },
        {
            "chunk_uid": 1,
            "source": "other.md",
            "chunk_id": 0,
            "text": "urban vegetation",
        },
    ]

    retriever = BM25Retriever(chunks)

    assert retriever.tokenized_chunks[0] == [
        "heat",
        "heat",
        "heat",
        "exposure",
    ]
    assert retriever.tokenized_chunks[0].count("heat") == 3
