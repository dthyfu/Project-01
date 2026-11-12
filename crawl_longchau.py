
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# =========================
# 1. CẤU HÌNH
# =========================

BASE_DIR = Path(__file__).resolve().parent
URL_FILE = BASE_DIR / "longchau_urls.txt"
DATA_DIR = BASE_DIR / "data"

OUTPUT_FILE = DATA_DIR / "longchau_diabetes.json"
LOG_FILE = DATA_DIR / "longchau_crawl_log.txt"

MAX_ARTICLES = 1321       # Chạy thử 10 bài trước
REQUEST_DELAY = 2       # Nghỉ 2 giây giữa các URL

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    )
}


# =========================
# 2. ĐỌC DANH SÁCH URL
# =========================

def load_urls():
    if not URL_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {URL_FILE}"
        )

    with open(URL_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Loại URL trùng, giữ nguyên thứ tự
    return list(dict.fromkeys(urls))


# =========================
# 3. TRÍCH XUẤT BÀI VIẾT
# =========================

def extract_article(url, session):
    response = session.get(
        url,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # Tìm vùng nội dung đã kiểm tra thành công
    article = soup.select_one(
        "div.posts-detail_posts-detail-container__GFMNQ"
    )

    if article is None:
        raise ValueError("Không tìm thấy vùng nội dung bài viết")

    # Xóa thành phần không thuộc nội dung cần thu thập
    for tag in article.select(
        "script, style, nav, footer, header, "
        "button, form, aside, iframe, svg"
    ):
        tag.decompose()

    parts = []

    for tag in article.find_all(["h2", "h3", "h4", "p", "li"]):
        text = tag.get_text(" ", strip=True)

        if not text:
            continue

        # Loại một số thành phần điều hướng thường gặp
        if text in {
            "Trang chủ",
            "Mặc định",
            "Lớn hơn",
            "Có thể bạn quan tâm"
        }:
            continue

        if not parts or text != parts[-1]:
            parts.append(text)

    # Loại đường dẫn điều hướng ở đầu bài nếu nằm riêng
    while parts and parts[0].startswith("Trang chủ"):
        parts.pop(0)

    # Cắt phần bài viết liên quan nếu tìm thấy tiêu đề
    for i, part in enumerate(parts):
        if part.strip().lower() == "có thể bạn quan tâm":
            parts = parts[:i]
            break

    content = "\n\n".join(parts).strip()

    if not title:
        raise ValueError("Không tìm thấy tiêu đề")

    if len(content) < 200:
        raise ValueError(
            f"Nội dung quá ngắn: {len(content)} ký tự"
        )

    # Tạo mã định danh ổn định theo URL
    article_id = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]

    return {
        "id": article_id,
        "title": title,
        "url": url,
        "source": urlparse(url).netloc,
        "domain": "diabetes",
        "content": content,
        "content_length": len(content),
        "crawled_at": datetime.now().isoformat(timespec="seconds")
    }


# =========================
# 4. ĐỌC DỮ LIỆU ĐÃ LƯU
# =========================

def load_existing():
    if not OUTPUT_FILE.exists():
        return []

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        print("Cảnh báo: JSON cũ không phải danh sách.")
        return []

    except (json.JSONDecodeError, OSError) as e:
        print("Không đọc được JSON cũ:", e)
        return []


# =========================
# 5. GHI LOG
# =========================

def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


# =========================
# 6. CHẠY CRAWLER
# =========================

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    urls = load_urls()
    existing = load_existing()

    # URL đã lưu thành công thì không tải lại
    saved = {item["url"]: item for item in existing if "url" in item}

    pending_urls = [url for url in urls if url not in saved]
    batch_urls = pending_urls[:MAX_ARTICLES]

    print("Tổng URL trong danh sách:", len(urls))
    print("Bài đã lưu trước đó:", len(saved))
    print("Bài sẽ thử trong lượt này:", len(batch_urls))

    if not batch_urls:
        print("Không còn URL mới cần xử lý.")
        return

    success = 0
    failed = 0

    with requests.Session() as session:
        for index, url in enumerate(batch_urls, start=1):
            print(f"\n[{index}/{len(batch_urls)}] Đang tải:")
            print(url)

            try:
                article = extract_article(url, session)

                # Lưu sau mỗi bài để hạn chế mất dữ liệu
                saved[url] = article
                success += 1

                print("Tiêu đề:", article["title"])
                print("Số ký tự:", article["content_length"])

                write_log(
                    f"SUCCESS | {url} | {article['title']}"
                )

            except Exception as e:
                failed += 1
                print("Lỗi:", str(e))

                write_log(f"FAILED | {url} | {str(e)}")

            finally:
                # Cập nhật JSON sau mỗi URL
                with open(
                    OUTPUT_FILE, "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        list(saved.values()),
                        f,
                        ensure_ascii=False,
                        indent=2
                    )

                if index < len(batch_urls):
                    time.sleep(REQUEST_DELAY)

    print("\n========== KẾT QUẢ ==========")
    print("Thành công trong lượt này:", success)
    print("Thất bại trong lượt này:", failed)
    print("Tổng bài đã lưu:", len(saved))
    print("File JSON:", OUTPUT_FILE)
    print("File log:", LOG_FILE)


if __name__ == "__main__":
    main()
