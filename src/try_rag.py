
from pathlib import Path

from loader import load_documents
from retriever import build_chunks
from bm25_retriever import BM25Retriever
from rag import answer_question


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

documents = load_documents(DATA_DIR)
chunks = build_chunks(documents)

retriever = BM25Retriever(chunks)

result = answer_question(
    query="What affects urban heat exposure?",
    retriever=retriever,
    top_k=3,
)

print("\nQuestion:")
print(result["query"])

print("\nAnswer:")
print(result["answer"])

print("\nSources:")
for source in result["sources"]:
    print(f"- {source['source']} | Chunk {source['chunk_id']}")
    print(source["text"])
