
import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

INPUT_FILE = "data/longchau_entities_normalized.json"
BATCH_SIZE = 200


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Chuyển dữ liệu thành các bản ghi quan hệ Chunk -> Entity
    rows = []

    for item in data:
        for entity in item.get("entities", []):
            rows.append({
                "chunk_id": item["chunk_id"],
                "name": entity["name"],
                "type": entity["type"],
                "original_name": entity["original_name"],
            })

    print("So chunk:", len(data))
    print("So luot lien ket thuc the:", len(rows))

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    )

    try:
        driver.verify_connectivity()

        # Đảm bảo mỗi thực thể có một node duy nhất theo tên chuẩn + loại
        driver.execute_query(
            """
            CREATE CONSTRAINT entity_identity_unique IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE (e.name, e.type) IS UNIQUE
            """,
            database_=DATABASE
        )

        query = """
        UNWIND $rows AS row

        MATCH (c:Chunk {chunk_id: row.chunk_id})

        MERGE (e:Entity {
            name: row.name,
            type: row.type
        })

        MERGE (c)-[r:MENTIONS]->(e)
        SET r.original_name = row.original_name
        """

        for batch in batches(rows, BATCH_SIZE):
            driver.execute_query(
                query,
                rows=batch,
                database_=DATABASE
            )

        # Thống kê số node thực thể
        result, _, _ = driver.execute_query(
            """
            MATCH (e:Entity)
            RETURN count(e) AS total
            """,
            database_=DATABASE
        )
        print("Tong node Entity:", result[0]["total"])

        # Thống kê số quan hệ MENTIONS
        result, _, _ = driver.execute_query(
            """
            MATCH (:Chunk)-[r:MENTIONS]->(:Entity)
            RETURN count(r) AS total
            """,
            database_=DATABASE
        )
        print("Tong quan he MENTIONS:", result[0]["total"])

        # Đếm thực thể theo loại
        result, _, _ = driver.execute_query(
            """
            MATCH (e:Entity)
            RETURN e.type AS type, count(e) AS total
            ORDER BY type
            """,
            database_=DATABASE
        )

        print("\nThong ke theo nhom:")
        for record in result:
            print(f"- {record['type']}: {record['total']}")

        print("\nDa nhap thuc the thanh cong!")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
