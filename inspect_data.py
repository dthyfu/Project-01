
import json

files = [
    "data/longchau_clean.json",
    "data/longchau_chunks.json",
]

for file_path in files:
    print(f"\n{'=' * 60}")
    print("FILE:", file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Kieu du lieu:", type(data).__name__)

    if isinstance(data, list):
        print("So ban ghi:", len(data))
        if data:
            print("Cac truong:", list(data[0].keys()))
            print("\nBan ghi dau tien:")
            print(json.dumps(data[0], ensure_ascii=False, indent=2)[:2000])

    elif isinstance(data, dict):
        print("Cac key:", list(data.keys()))
        for key, value in data.items():
            if isinstance(value, list):
                print(f"Key '{key}': {len(value)} ban ghi")
                if value:
                    print("Cac truong ban ghi dau tien:", list(value[0].keys()))
                break
