
import time
import requests
import xml.etree.ElementTree as ET

BASE_URL = "https://nhathuoclongchau.com.vn"

SITEMAPS = [
    f"{BASE_URL}/sitemap_benh.xml",
    *[
        f"{BASE_URL}/sitemap_baiviet{i}.xml"
        for i in range(1, 18)
    ],
]

KEYWORDS = [
    "tieu-duong",
    "dai-thao-duong",
    "duong-huyet",
    "insulin",
]

session = requests.Session()
session.headers.update({
    "User-Agent": "HybridRAG-Student/1.0"
})


def get_urls(sitemap_url):
    response = session.get(sitemap_url, timeout=20)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    urls = [
        element.text.strip()
        for element in root.iter()
        if element.tag.endswith("loc") and element.text
    ]

    return root.tag.split("}")[-1], urls


def main():
    matches = []
    errors = []

    for sitemap_url in SITEMAPS:
        try:
            sitemap_type, urls = get_urls(sitemap_url)

            # Nếu sitemap trỏ tiếp đến sitemap con
            if sitemap_type == "sitemapindex":
                print(f"Sitemap index: {sitemap_url}")
                print("Sitemap này còn chứa sitemap con.")

            for url in urls:
                if any(keyword in url.lower() for keyword in KEYWORDS):
                    matches.append(url)

            print(f"Đã kiểm tra: {sitemap_url} ({len(urls)} URL)")

        except (requests.RequestException, ET.ParseError) as e:
            errors.append((sitemap_url, str(e)))
            print(f"Không đọc được sitemap: {sitemap_url}")

        # Hạn chế gửi yêu cầu liên tục
        time.sleep(0.5)

    matches = sorted(set(matches))

    print("\n===== KẾT QUẢ =====")
    print("Số URL phù hợp:", len(matches))

    for url in matches[:50]:
        print(url)

    with open("longchau_urls.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(matches))

    print("\nĐã lưu URL vào longchau_urls.txt")

    if errors:
        print("\nSố sitemap gặp lỗi:", len(errors))
        for url, error in errors:
            print(url, "->", error)


if __name__ == "__main__":
    main()
