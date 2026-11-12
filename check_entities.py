
import json
from collections import Counter, defaultdict

INPUT_FILE = "data/longchau_entities.json"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Đếm số lần xuất hiện và số chunk chứa mỗi thực thể
    mention_counts = Counter()
    chunk_counts = Counter()
    types_by_name = defaultdict(set)

    for item in data:
        seen_in_chunk = set()

        for entity in item.get("entities", []):
            name = entity["name"].strip().lower()
            entity_type = entity["type"]
            key = (entity_type, name)

            mention_counts[key] += 1
            types_by_name[name].add(entity_type)
            seen_in_chunk.add(key)

        for key in seen_in_chunk:
            chunk_counts[key] += 1

    print("=== THONG KE THUC THE ===")
    print("So ten thuc the duy nhat:", len(mention_counts))

    print("\nSo ten duy nhat theo nhom:")
    type_names = defaultdict(set)

    for entity_type, name in mention_counts:
        type_names[entity_type].add(name)

    for entity_type, names in sorted(type_names.items()):
        print(f"- {entity_type}: {len(names)} ten")

    print("\n=== 20 THUC THE CO NHIEU LUOT NHAN DIEN NHAT ===")
    for (entity_type, name), count in mention_counts.most_common(20):
        print(
            f"- {name} [{entity_type}]: "
            f"{count} luot, {chunk_counts[(entity_type, name)]} chunk"
        )

    print("\n=== TEN XUAT HIEN O NHIEU NHOM ===")
    for name, types in sorted(types_by_name.items()):
        if len(types) > 1:
            print(f"- {name}: {', '.join(sorted(types))}")


if __name__ == "__main__":
    main()
