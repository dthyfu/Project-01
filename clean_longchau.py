
import json
import re
from pathlib import Path

# =========================
# 1. CẤU HÌNH
# =========================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "longchau_diabetes.json"
OUTPUT_FILE = BASE_DIR / "data" / "longchau_clean.json"


# =========================
# 2. LÀM SẠCH NỘI DUNG
# =========================

def clean_content(content):
    if not isinstance(content, str):
        return ""

    # Chuẩn hóa ký tự xuống dòng
    content = content.replace("\r\n", "\n")
    content = content.replace("\r", "\n")

    # Loại phần danh sách bài viết liên quan ở cuối bài
    import re

    related_pattern = (
        r"(?im)^\s*(?:"
        r"các bài viết liên quan|"
        r"bài viết liên quan|"
        r"có thể bạn quan tâm"
        r")\s*:?\s*$"
    )

    match = re.search(related_pattern, content)

    if match:
        content = content[:match.start()]

    # Chuẩn hóa khoảng trắng và loại dòng trống
    lines = []

    for line in content.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    content = "\n\n".join(lines)

    # Loại các đoạn trùng nhau liên tiếp
    paragraphs = content.split("\n\n")
    cleaned_paragraphs = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if cleaned_paragraphs and paragraph == cleaned_paragraphs[-1]:
            continue

        cleaned_paragraphs.append(paragraph)

    return "\n\n".join(cleaned_paragraphs).strip()



# =========================
# 3. XỬ LÝ DỮ LIỆU
# =========================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not isinstance(articles, list):
        raise ValueError("Dữ liệu JSON phải là một danh sách.")

    cleaned_articles = []

    for article in articles:
        cleaned = article.copy()

        # Chỉ làm sạch nội dung; không sửa dữ liệu gốc
        cleaned["content"] = clean_content(
            article.get("content", "")
        )
        cleaned["content_length"] = len(cleaned["content"])

        cleaned_articles.append(cleaned)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            cleaned_articles,
            f,
            ensure_ascii=False,
            indent=2
        )

    lengths = [
        len(article["content"])
        for article in cleaned_articles
    ]

    print("========== KẾT QUẢ ==========")
    print("Số bài đầu vào:", len(articles))
    print("Số bài đầu ra:", len(cleaned_articles))
    print("Số bài có nội dung rỗng:", sum(
        length == 0 for length in lengths
    ))
    print("Độ dài nội dung trung bình:", round(
        sum(lengths) / max(len(lengths), 1)
    ))
    print("File đầu ra:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
