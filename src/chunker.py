
from pathlib import Path


def split_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 30,
) -> list[str]:
    """Split text near word boundaries with overlapping context."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        next_start = max(0, end - overlap)

        # Move the next start to a word boundary.
        space_at = text.find(" ", next_start, end)
        if space_at != -1:
            next_start = space_at + 1

        start = next_start

    return chunks


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sample_path = project_root / "data" / "sample.md"

    text = sample_path.read_text(encoding="utf-8")
    chunks = split_text(text, chunk_size=200, overlap=30)

    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)
