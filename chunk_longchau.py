
import json
import re
from pathlib import Path

# Thư mục gốc của project
BASE_DIR = Path(__file__).resolve().parent

# File dữ liệu đã làm sạch
INPUT_FILE = BASE_DIR / "data" / "longchau_clean.json"

# File chứa các chunks sau khi chia
OUTPUT_FILE = BASE_DIR / "data" / "longchau_chunks.json"

# Cấu hình chia chunks (tính theo ký tự)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def normalize_content(text):
    """Chuẩn hóa nội dung nhưng giữ lại ranh giới đoạn văn."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Chia văn bản thành chunks.
    Ưu tiên ranh giới đoạn văn, sau đó đến câu và khoảng trắng.
    """
    text = normalize_content(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap phải >= 0 và nhỏ hơn chunk_size."
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # Chỉ tìm điểm cắt trong nửa sau của chunk
            # để tránh tạo chunk quá ngắn.
            min_cut = start + int(chunk_size * 0.6)

            # Ưu tiên cắt tại ranh giới đoạn văn.
            paragraph_pos = text.rfind("\n\n", min_cut, end)

            if paragraph_pos != -1:
                end = paragraph_pos
            else:
                # Nếu không có ranh giới đoạn văn,
                # tìm dấu kết thúc câu gần cuối chunk.
                sentence_positions = [
                    match.end()
                    for match in re.finditer(
                        r'[.!?…]["”’)\]]*\s+',
                        text[min_cut:end]
                    )
                ]

                if sentence_positions:
                    end = min_cut + sentence_positions[-1]
                else:
                    # Cuối cùng mới cắt tại khoảng trắng.
                    space_pos = text.rfind(" ", min_cut, end)

                    if space_pos != -1:
                        end = space_pos

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Tạo overlap, ưu tiên bắt đầu ở ranh giới đoạn
        # hoặc câu để giữ ngữ cảnh.
        next_start = max(start + 1, end - overlap)
        search_start = next_start
        search_end = min(end, next_start + 100)

        paragraph_start = text.find("\n\n", search_start, search_end)

        if paragraph_start != -1:
            next_start = paragraph_start + 2
        else:
            sentence_matches = list(
                re.finditer(
                    r'[.!?…]["”’)\]]*\s+',
                    text[search_start:search_end]
                )
            )

            if sentence_matches:
                next_start = (
                    search_start + sentence_matches[0].end()
                )

        # Phòng trường hợp không tiến được trong văn bản.
        start = max(next_start, start + 1)

    return chunks


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        articles = json.load(file)

    all_chunks = []
    chunk_id = 1

    for article in articles:
        content = article.get("content", "")
        article_chunks = split_text(content)

        for index, chunk_text in enumerate(article_chunks, start=1):
            all_chunks.append({
                "chunk_id": chunk_id,
                "article_id": article.get("id"),
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "chunk_index": index,
                "content": chunk_text,
            })

            chunk_id += 1

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("========== KẾT QUẢ CHIA CHUNKS ==========")
    print("Số bài viết đầu vào:", len(articles))
    print("Tổng số chunks:", len(all_chunks))
    print(
        "Số chunks trung bình mỗi bài:",
        round(len(all_chunks) / max(len(articles), 1), 2),
    )
    print(
        "Số chunks có nội dung rỗng:",
        sum(not item["content"].strip() for item in all_chunks),
    )
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("File đầu ra:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
