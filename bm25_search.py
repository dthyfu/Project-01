
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

# Đường dẫn dữ liệu
BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "longchau_chunks.json"


def tokenize(text):
    """
    Chuẩn hóa văn bản và tách thành các từ.
    Tạm thời dùng cách tách đơn giản theo khoảng trắng.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {CHUNKS_FILE}"
        )

    chunks = load_chunks()

    # Chuẩn bị nội dung để BM25 lập chỉ mục
    tokenized_corpus = [
        tokenize(chunk.get("content", ""))
        for chunk in chunks
    ]

    # Khởi tạo bộ tìm kiếm BM25
    bm25 = BM25Okapi(tokenized_corpus)

    print("Đã nạp dữ liệu BM25 thành công!")
    print("Tổng số chunks:", len(chunks))

    # Nhập câu hỏi để thử tìm kiếm
    query = input("\nNhập câu hỏi cần tìm: ").strip()

    if not query:
        print("Bạn chưa nhập câu hỏi.")
        return

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Lấy tối đa 5 chunks có điểm cao nhất
    top_k = min(5, len(chunks))
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    print("\n========== KẾT QUẢ BM25 ==========")

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]

        print(f"\n--- Kết quả {rank} ---")
        print("Điểm BM25:", round(float(scores[index]), 4))
        print("Tiêu đề:", chunk.get("title", ""))
        print("Chunk ID:", chunk.get("chunk_id"))
        print("Nội dung:", chunk.get("content", "")[:500])
        print("Nguồn:", chunk.get("url", ""))


if __name__ == "__main__":
    main()
