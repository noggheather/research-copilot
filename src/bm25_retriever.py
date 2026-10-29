
import re

from rank_bm25 import BM25Okapi


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text while preserving repeated words for BM25."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        # BM25 使用列表保存词语，保留重复词及其词频
        self.tokenized_chunks = [
            tokenize_bm25(chunk["text"])
            for chunk in chunks
        ]

        # 空文本块不参与 BM25 索引
        self.valid_indices = [
            i
            for i, tokens in enumerate(self.tokenized_chunks)
            if tokens
        ]

        self.bm25 = None

        if self.valid_indices:
            corpus = [
                self.tokenized_chunks[i]
                for i in self.valid_indices
            ]
            self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = tokenize_bm25(query)

        if not query_tokens or self.bm25 is None:
            return []

        scores = self.bm25.get_scores(query_tokens)
        results = []

        for index, score in zip(self.valid_indices, scores):
            chunk = self.chunks[index]

            # 不返回完全没有查询词匹配的文本块
            if not (set(query_tokens) & set(self.tokenized_chunks[index])):
                continue

            results.append({
                **chunk,
                "score": float(score),
            })

        results.sort(
            key=lambda item: (
                -item["score"],
                item["source"],
                item["chunk_id"],
            )
        )

        return results[:top_k]
