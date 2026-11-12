
import json
import os
import re
import unicodedata

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

ENTITY_FILE = "data/longchau_entities_normalized.json"


def normalize_text(text):
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def load_entity_dictionary():
    """Đọc các thực thể đã chuẩn hóa."""
    with open(ENTITY_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    entities = {}

    for item in data:
        for entity in item.get("entities", []):
            key = (entity["type"], entity["name"])
            entities[key] = {
                "name": entity["name"],
                "type": entity["type"],
            }

    return list(entities.values())



def find_query_entities(question, dictionary):
    """Tim thuc the, co ho tro ten dong nghia."""

    normalized_question = normalize_text(question)

    # Chuan hoa cach goi pho bien trong cau hoi
    aliases = {
        "bệnh tiểu đường": "đái tháo đường",
        "tiểu đường type 1": "đái tháo đường type 1",
        "tiểu đường tuýp 1": "đái tháo đường type 1",
        "tiểu đường type 2": "đái tháo đường type 2",
        "tiểu đường tuýp 2": "đái tháo đường type 2",
    }

    for alias, canonical in aliases.items():
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        normalized_question = re.sub(
            pattern,
            canonical,
            normalized_question
        )

    found = {}

    for entity in dictionary:
        name = normalize_text(entity["name"])
        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"

        if re.search(pattern, normalized_question):
            key = (entity["type"], entity["name"])
            found[key] = entity

    return list(found.values())



def graph_search(driver, question, top_k=5):
    dictionary = load_entity_dictionary()
    matched_entities = find_query_entities(question, dictionary)

    if not matched_entities:
        print(
            "\nKhong tim thay thuc the trong cau hoi. "
            "Hay thu dien dat ro hon."
        )
        return []

    print("\nThuc the tim thay:")
    for entity in matched_entities:
        print(f"- [{entity['type']}] {entity['name']}")

    
    query = """
    UNWIND $entities AS item

    MATCH (e:Entity {
        name: item.name,
        type: item.type
    })

    MATCH (c:Chunk)-[:MENTIONS]->(e)
    MATCH (a:Article)-[:HAS_CHUNK]->(c)

    WITH c, a,
         collect(DISTINCT e.name) AS matched_entities,
         count(DISTINCT e) AS matched_count

    OPTIONAL MATCH (c)-[:MENTIONS]->(related:Entity)

    WITH c, a, matched_entities, matched_count,
         collect(DISTINCT CASE
             WHEN related.type = $target_type
             THEN related.name
             ELSE null
         END) AS target_entities

    RETURN
        c.chunk_id AS chunk_id,
        c.content AS content,
        c.title AS title,
        c.url AS url,
        matched_entities,
        matched_count,
        [x IN target_entities WHERE x IS NOT NULL]
            AS target_entities,
        size([x IN target_entities WHERE x IS NOT NULL])
            AS target_count

    ORDER BY target_count DESC, matched_count DESC, chunk_id ASC
    LIMIT $top_k
    """


    rows = [
        {
            "name": entity["name"],
            "type": entity["type"],
        }
        for entity in matched_entities
    ]

    
    question_lower = normalize_text(question)

    if any(word in question_lower for word in [
        "triệu chứng", "dấu hiệu", "biểu hiện"
    ]):
        target_type = "Symptom"
    elif any(word in question_lower for word in [
        "biến chứng", "hậu quả"
    ]):
        target_type = "Complication"
    elif any(word in question_lower for word in [
        "yếu tố nguy cơ", "nguy cơ mắc"
    ]):
        target_type = "RiskFactor"
    else:
        target_type = ""

    records, _, _ = driver.execute_query(
        query,
        entities=rows,
        target_type=target_type,
        top_k=top_k,
        database_=DATABASE,
    )


    return [record.data() for record in records]


def main():
    question = input("Nhap cau hoi: ").strip()

    if not question:
        print("Cau hoi khong duoc de trong.")
        return

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD),
    )

    try:
        driver.verify_connectivity()

        results = graph_search(
            driver,
            question,
            top_k=5,
        )

        print(f"\nSo ket qua Graph Search: {len(results)}")

        for index, result in enumerate(results, start=1):
            print("\n" + "=" * 60)
            print(f"KET QUA {index}")
            print("Chunk ID:", result["chunk_id"])
            print("Tieu de:", result["title"])
            print("Thuc the khop:", result["matched_entities"])
            print("Noi dung:", (result["content"] or "")[:1000])
            print("Nguon:", result["url"])

    except Exception as error:
        print("Graph Search gap loi:", error)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
