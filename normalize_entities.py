
import json
import re
import unicodedata

INPUT_FILE = "data/longchau_entities.json"
OUTPUT_FILE = "data/longchau_entities_normalized.json"

# Chỉ gộp các cách viết tương đương đã xác định.
# Không gộp các bệnh có ý nghĩa khác nhau.
ALIASES = {
    "bệnh tiểu đường": "đái tháo đường",
    "tiểu đường type 1": "đái tháo đường type 1",
    "tiểu đường tuýp 1": "đái tháo đường type 1",
    "tiểu đường type 2": "đái tháo đường type 2",
    "tiểu đường tuýp 2": "đái tháo đường type 2",
    "đi tiểu nhiều": "tiểu nhiều",
}


def normalize_text(text):
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    normalized_data = []
    unique_entities = set()

    for item in data:
        normalized_entities = []

        for entity in item.get("entities", []):
            original_name = entity["name"]
            normalized_name = normalize_text(original_name)
            canonical_name = ALIASES.get(
                normalized_name,
                normalized_name
            )

            normalized_entities.append({
                "name": canonical_name,
                "type": entity["type"],
                "original_name": original_name,
            })

            unique_entities.add(
                (entity["type"], canonical_name)
            )

        # Loại bỏ trùng cùng thực thể trong một chunk
        deduplicated = {}
        for entity in normalized_entities:
            key = (entity["type"], entity["name"])
            if key not in deduplicated:
                deduplicated[key] = entity

        normalized_data.append({
            **item,
            "entities": list(deduplicated.values()),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            normalized_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Da chuan hoa xong!")
    print("So chunk:", len(normalized_data))
    print("So thuc the duy nhat sau chuan hoa:",
          len(unique_entities))
    print("File:", OUTPUT_FILE)

    print("\nDanh sach thuc the:")
    for entity_type, name in sorted(unique_entities):
        print(f"- [{entity_type}] {name}")


if __name__ == "__main__":
    main()
