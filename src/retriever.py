
from pathlib import Path
import re

from loader import load_documents
from chunker import split_text
"""`读取文件夹里所有文档` → `把长文档切成小段（chunk）`并编号 → `你输入问题` → `程序用关键词比对，找出和问题词语重合最多的文本片段` → 打印结果"""

def tokenize(text: str) -> set[str]:    #将文本转成小写英文词集合
    """Convert text into lowercase word tokens."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))   #正则提取所有英文字母 + 数字，过滤掉标点、换行、空格



def build_chunks(documents: list[dict]) -> list[dict]:
    all_chunks = []

    for document in documents:
        text_chunks = split_text(
            document["content"],
            chunk_size=200,
            overlap=30,
        )

        for chunk_id, text in enumerate(text_chunks):
            all_chunks.append({
                "chunk_uid": len(all_chunks),
                "source": document["source"],
                "chunk_id": chunk_id,
                "text": text,
            })


    return all_chunks


def search_chunks(          #计算关键词匹配分数并排序
    query: str,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Rank chunks by the number of query terms they contain."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_tokens = tokenize(query)      #把用户输入的问题，提取单词集合。用户提问：`What is RAG?` → `{"what", "is", "rag"}`

    if not query_tokens:
        return []

    results = []

    for chunk in chunks:
        chunk_tokens = tokenize(chunk["text"])
        matched_tokens = query_tokens & chunk_tokens

        if matched_tokens:
            results.append({
                **chunk,
                "score": len(matched_tokens),
                "matched_terms": sorted(matched_tokens),
            })

    results.sort(
        key=lambda item: (
            -item["score"],
            item["source"],
            item["chunk_id"],
        )
    )

    return results[:top_k]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_folder = project_root / "data"

    documents = load_documents(str(data_folder))
    chunks = build_chunks(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")

    query = input("\nEnter your question: ")
    results = search_chunks(query, chunks, top_k=3)

    if not results:
        print("\nNo matching passages found.")
    else:
        print("\nSearch results:")

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Source: {result['source']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Score: {result['score']}")
            print(f"Matched terms: {result['matched_terms']}")
            print(f"Text: {result['text']}")
