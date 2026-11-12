
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "longchau_chunks.json"
VECTOR_FILE = BASE_DIR / "data" / "longchau_vectors.npy"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Đang tải mô hình embedding...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk.get("content", "")
        for chunk in chunks
    ]

    print("Đang chuyển chunks thành vector...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(VECTOR_FILE, embeddings)

    print("\n========== KẾT QUẢ VECTOR ==========")
    print("Số chunks:", len(chunks))
    print("Kích thước vector:", embeddings.shape)
    print("File vector:", VECTOR_FILE)

    # Thử tìm kiếm bằng câu hỏi
    query = input("\nNhập câu hỏi cần tìm: ").strip()

    if not query:
        print("Bạn chưa nhập câu hỏi.")
        return

    query_vector = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Vector đã chuẩn hóa nên tích vô hướng là cosine similarity
    scores = np.dot(embeddings, query_vector)

    top_k = min(5, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    print("\n========== KẾT QUẢ VECTOR SEARCH ==========")

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]

        print(f"\n--- Kết quả {rank} ---")
        print("Điểm tương đồng:", round(float(scores[index]), 4))
        print("Tiêu đề:", chunk.get("title", ""))
        print("Chunk ID:", chunk.get("chunk_id"))
        print("Nội dung:", chunk.get("content", "")[:500])
        print("Nguồn:", chunk.get("url", ""))


if __name__ == "__main__":
    main()
