
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Đọc thông tin kết nối từ file .env
load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not URI or not USERNAME or not PASSWORD:
    raise ValueError(
        "Thiếu thông tin Neo4j. Hãy kiểm tra file .env"
    )

try:
    with GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    ) as driver:

        # Kiểm tra khả năng kết nối
        driver.verify_connectivity()
        print("Ket noi Neo4j thanh cong!")

        # Kiểm tra database có node và relationship chưa
        records, summary, keys = driver.execute_query(
            """
            MATCH (n)
            RETURN count(n) AS total_nodes
            """,
            database_=DATABASE
        )

        print("So node hien tai:", records[0]["total_nodes"])

        records, summary, keys = driver.execute_query(
            """
            MATCH ()-[r]->()
            RETURN count(r) AS total_relationships
            """,
            database_=DATABASE
        )

        print(
            "So relationship hien tai:",
            records[0]["total_relationships"]
        )

except Exception as e:
    print("Ket noi that bai!")
    print("Chi tiet loi:", e)
