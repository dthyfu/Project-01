
import json
import re
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent

URL_FILE = BASE_DIR / "longchau_urls.txt"
JSON_FILE = BASE_DIR / "data" / "longchau_diabetes.json"
OUTPUT_FILE = BASE_DIR / "longchau_urls_selected.txt"

TARGET_TOTAL = 100

# Các nhóm chủ đề ưu tiên.
TOPIC_KEYWORDS = {
    "benh-tieu-duong": 5,
    "dai-thao-duong": 5,
    "tieu-duong": 5,
    "duong-huyet": 4,
    "insulin": 3,
    "trieu-chung": 3,
    "nguyen-nhan": 2,
    "chan-doan": 3,
    "dieu-tri": 3,
    "bien-chung": 3,
    "ha-duong-huyet": 4,
    "tang-duong-huyet": 4,
    "che-do-an": 3,
    "thuc-pham": 2,
    "dinh-duong": 2,
    "phong-ngua": 2,
    "chi-so": 2,
    "xet-nghiem": 2,
    "thuoc": 2,
    "glucose": 2,
}


def normalize_url(url):
    return url.strip().rstrip("/")


# Đọc danh sách URL nguồn.
with open(URL_FILE, "r", encoding="utf-8") as f:
    urls = list(dict.fromkeys(
        normalize_url(line)
        for line in f
        if line.strip()
    ))

# Đọc các bài đã crawl để không chọn trùng.
saved_urls = set()

if JSON_FILE.exists():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    saved_urls = {
        normalize_url(article["url"])
        for article in articles
        if article.get("url")
    }

# Chấm điểm URL dựa trên từ khóa trong đường dẫn.
ranked_urls = []

for url in urls:
    if normalize_url(url) in saved_urls:
        continue

    slug = urlparse(url).path.lower()
    slug = re.sub(r"\.html$", "", slug)

    score = sum(
        weight
        for keyword, weight in TOPIC_KEYWORDS.items()
        if keyword in slug
    )

    # Chỉ giữ các URL có dấu hiệu liên quan đến chủ đề.
    if score > 0:
        ranked_urls.append((score, url))

# Điểm cao hơn được ưu tiên.
ranked_urls.sort(key=lambda item: item[0], reverse=True)

# Giữ tổng khoảng 100 bài, tính cả các bài đã có.
remaining = max(0, TARGET_TOTAL - len(saved_urls))
selected = [url for score, url in ranked_urls[:remaining]]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for url in selected:
        f.write(url + "\n")

print(f"Tổng URL nguồn: {len(urls)}")
print(f"Bài đã crawl: {len(saved_urls)}")
print(f"URL phù hợp còn lại: {len(ranked_urls)}")
print(f"URL được chọn thêm: {len(selected)}")
print(f"Dự kiến tổng số bài: {len(saved_urls) + len(selected)}")
print(f"Danh sách đã lưu tại: {OUTPUT_FILE}")
