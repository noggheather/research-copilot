from pathlib import Path
import sys
import pytest

# 将 src 目录加入 Python 的模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from retriever import build_chunks, search_chunks


@pytest.fixture
def chunks():
    """构造一组测试用文本，不依赖真实数据文件。"""
    documents = [
        {
            "source": "sample.md",
            "content": (
                "Thermal exposure is related to high temperature. "
                "Urban heat can affect human health."
            ),
        },
        {
            "source": "sample2.md",
            "content": (
                "Green spaces and vegetation can help reduce urban heat."
            ),
        },
    ]
    return build_chunks(documents)


def test_relevant_query_returns_results(chunks):
    """相关查询应该命中文本。"""
    results = search_chunks("thermal exposure", chunks)

    assert len(results) > 0
    assert results[0]["score"] > 0


def test_unrelated_query_returns_empty_list(chunks):
    """完全无关的查询应该返回空列表。"""
    results = search_chunks("quantum spaceship", chunks)

    assert results == []


def test_blank_query_returns_empty_list(chunks):
    """空白查询应该返回空列表。"""
    results = search_chunks("   ", chunks)

    assert results == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises_error(chunks, top_k):
    """top_k 小于等于 0 时应该报错。"""
    with pytest.raises(ValueError):
        search_chunks("thermal exposure", chunks, top_k=top_k)


def test_every_result_has_source(chunks):
    """每条检索结果都应该保留来源文件信息。"""
    results = search_chunks("heat", chunks)

    assert len(results) > 0

    for result in results:
        assert "source" in result
        assert result["source"]
