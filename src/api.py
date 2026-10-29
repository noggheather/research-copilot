from pathlib import Path
from typing import Literal  # 新增：限制 method 的可选值

from fastapi import FastAPI
from pydantic import BaseModel, Field

from loader import load_documents
from retriever import build_chunks, search_chunks
from bm25_retriever import BM25Retriever  # 新增

from fastapi import HTTPException
from rag import answer_question
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# 加载文档并构建文本块
documents = load_documents(DATA_DIR)
chunks = build_chunks(documents)

# 新增：创建 BM25 检索器
bm25_retriever = BM25Retriever(chunks)

app = FastAPI(
    title="Research Copilot",
    description="A research document retrieval API",
    version="0.2.0",
)


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Search query"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=20
    )

    # 新增：允许用户选择检索算法,Literal 用来限制变量只能取某几个指定值。
    method: Literal["keyword", "bm25"] = Field(
        default="keyword",
        description="Retrieval method"
    )


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    return {
        "count": len(documents),
        "documents": [
            {
                "source": doc["source"],
                "char_count": doc["char_count"],
            }
            for doc in documents
        ],
    }


@app.post("/search")
def search(request: SearchRequest):

    # 新增：根据 method 选择不同检索算法
    if request.method == "bm25":
        results = bm25_retriever.search(
            request.query,
            top_k=request.top_k
        )

    else:
        results = search_chunks(
            request.query,
            chunks,
            top_k=request.top_k
        )

    return {
        "query": request.query,
        "method": request.method,  # 新增：返回使用的算法
        "result_count": len(results),
        "results": results,
    }


@app.post("/ask")
def ask(request: AskRequest):
    try:
        result = answer_question(
            query=request.query,
            retriever=bm25_retriever,
            top_k=request.top_k,
        )
        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="Failed to communicate with the local Ollama service.",
        ) from exc
