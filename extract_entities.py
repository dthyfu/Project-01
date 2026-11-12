
import json
import re
import unicodedata
from collections import Counter

INPUT_FILE = "data/longchau_chunks.json"
OUTPUT_FILE = "data/longchau_entities.json"

# Từ điển thuật ngữ ban đầu.
# Có thể mở rộng sau khi kiểm tra kết quả thực tế.
ENTITY_DICTIONARY = {
    "Disease": [
        "bệnh tiểu đường",
        "đái tháo đường",
        "tiểu đường type 1",
        "tiểu đường type 2",
        "tiểu đường tuýp 1",
        "tiểu đường tuýp 2",
        "hạ đường huyết",
        "tăng đường huyết",
        "tiền tiểu đường",
        "bệnh tiểu đường thai kỳ",
    ],
    "Symptom": [
        "khát nước",
        "khát nhiều",
        "tiểu nhiều",
        "đi tiểu nhiều",
        "đói nhiều",
        "đói cồn cào",
        "mệt mỏi",
        "sụt cân",
        "sụt cân không rõ nguyên nhân",
        "nhìn mờ",
        "mờ mắt",
        "tê bì chân tay",
        "tê bì tay chân",
        "vết thương lâu lành",
        "ngứa da",
        "khô miệng",
        "chóng mặt",
        "run rẩy tay chân",
        "ra mồ hôi lạnh",
        "lo lắng",
    ],
    "Complication": [
        "biến chứng thần kinh",
        "bệnh võng mạc tiểu đường",
        "bệnh thận tiểu đường",
        "suy thận",
        "bệnh tim mạch",
        "đột quỵ",
        "loét bàn chân",
        "bàn chân tiểu đường",
        "tổn thương thần kinh",
        "tổn thương thận",
        "tổn thương mắt",
    ],
    "RiskFactor": [
        "thừa cân",
        "béo phì",
        "ít vận động",
        "tiền sử gia đình",
        "huyết áp cao",
        "tăng huyết áp",
        "rối loạn mỡ máu",
        "tuổi cao",
        "hút thuốc lá",
    ],
}


def normalize_text(text):
    """Chuẩn hóa chữ thường và khoảng trắng."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def find_terms(text, entity_type, terms):
    """Tìm thuật ngữ nguyên cụm, tránh khớp bên trong từ dài hơn."""
    normalized = normalize_text(text)
    found = []

    for term in terms:
        term_normalized = normalize_text(term)

        pattern = (
            r"(?<!\w)"
            + re.escape(term_normalized)
            + r"(?!\w)"
        )

        if re.search(pattern, normalized):
            found.append({
                "name": term,
                "type": entity_type,
            })

    return found


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    results = []
    type_counts = Counter()

    for chunk in chunks:
        text = chunk.get("content", "")
        entities_by_key = {}

        for entity_type, terms in ENTITY_DICTIONARY.items():
            matches = find_terms(text, entity_type, terms)

            for entity in matches:
                key = (entity["type"], normalize_text(entity["name"]))
                entities_by_key[key] = entity

        entities = list(entities_by_key.values())

        for entity in entities:
            type_counts[entity["type"]] += 1

        results.append({
            "chunk_id": chunk["chunk_id"],
            "article_id": chunk["article_id"],
            "title": chunk["title"],
            "url": chunk["url"],
            "entities": entities,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    chunks_with_entities = sum(
        1 for item in results if item["entities"]
    )

    print("Da trich xuat xong!")
    print("Tong chunk:", len(chunks))
    print("Chunk co thuc the:", chunks_with_entities)
    print("Chunk khong co thuc the:",
          len(chunks) - chunks_with_entities)
    print("\nSo luot nhan dien theo nhom:")

    for entity_type, count in type_counts.items():
        print(f"- {entity_type}: {count}")

    print("\nFile ket qua:", OUTPUT_FILE)

    print("\nMot so ket qua mau:")
    shown = 0

    for item in results:
        if item["entities"]:
            print("\nChunk:", item["chunk_id"])
            print("Bai viet:", item["title"])
            print("Thuc the:", item["entities"])

            shown += 1
            if shown == 5:
                break


if __name__ == "__main__":
    main()
