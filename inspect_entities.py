
import json

path = "data/longchau_entities_normalized.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Kieu du lieu:", type(data).__name__)

if isinstance(data, dict):
    print("Cac key:", list(data.keys()))
    for key, value in data.items():
        print(f"\nKey: {key}")
        print("Kieu:", type(value).__name__)
        if isinstance(value, list) and value:
            print("Phan tu dau tien:")
            print(json.dumps(value[0], ensure_ascii=False, indent=2))
            break

elif isinstance(data, list) and data:
    print("Phan tu dau tien:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))
