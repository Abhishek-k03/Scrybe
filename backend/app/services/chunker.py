def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[dict] = []
    step = chunk_size - overlap
    i = 0
    idx = 0
    n = len(text)
    while i < n:
        piece = text[i : i + chunk_size].strip()
        if piece:
            chunks.append({"text": piece, "chunk_index": idx})
            idx += 1
        if i + chunk_size >= n:
            break
        i += step
    return chunks
