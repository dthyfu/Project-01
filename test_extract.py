
import requests
from bs4 import BeautifulSoup

URL = "https://nhathuoclongchau.com.vn/bai-viet/10-bieu-hien-va-cach-xu-ly-ha-duong-huyet-kip-thoi-42197.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    )
}


def extract_article(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Lấy tiêu đề
    title_tag = soup.find("h1")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # Tìm vùng chứa bài viết
    article = soup.select_one(
        "div.posts-detail_posts-detail-container__GFMNQ"
    )

    if article is None:
        raise ValueError("Không tìm thấy vùng nội dung bài viết.")

    # Xóa những thành phần không cần thiết
    for tag in article.select(
        "script, style, nav, footer, header, button, "
        "form, aside, iframe, svg"
    ):
        tag.decompose()

    # Trích xuất tiêu đề phụ và các đoạn văn
    parts = []

    for tag in article.find_all(["h2", "h3", "h4", "p", "li"]):
        text = tag.get_text(" ", strip=True)

        if not text:
            continue

        if text in {"Trang chủ", "Mặc định", "Lớn hơn"}:
            continue

        if not parts or text != parts[-1]:
            parts.append(text)

    # Loại bỏ đường dẫn điều hướng ở đầu bài
    while parts and (
        parts[0] == "Trang chủ"
        or parts[0].startswith("Trang chủ /")
    ):
        parts.pop(0)

    if parts and parts[0] == "/":
        parts.pop(0)

    # Loại bỏ danh sách bài viết liên quan ở cuối
    for i, part in enumerate(parts):
        if part.strip().lower() == "có thể bạn quan tâm":
            parts = parts[:i]
            break

    content = "\n\n".join(parts)

    if len(content) < 200:
        raise ValueError(
            f"Nội dung trích xuất quá ngắn: {len(content)} ký tự."
        )

    return {
        "title": title,
        "url": url,
        "content": content
    }



if __name__ == "__main__":
    with open("longchau_urls.txt", "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()][:5]

    for i, url in enumerate(urls, start=1):
        print(f"\n{'=' * 60}")
        print(f"Bài {i}/5")
        print("URL:", url)

        try:
            data = extract_article(url)
            print("Tiêu đề:", data["title"])
            print("Độ dài:", len(data["content"]), "ký tự")
            print("Nội dung đầu:", data["content"][:250].replace("\n", " "))

        except Exception as e:
            print("Lỗi:", e)
