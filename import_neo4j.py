
import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Mỗi lô chỉ gửi 200 bản ghi
BATCH_SIZE = 200


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main():
    articles = load_json("data/longchau_clean.json")
    chunks = load_json("data/longchau_chunks.json")

    print(f"So bai viet: {len(articles)}")
    print(f"So chunk: {len(chunks)}")

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    )

    try:
        driver.verify_connectivity()
        print("\nDa ket noi Neo4j!")

        # 1. Tao rang buoc duy nhat cho ID
        driver.execute_query(
            """
            CREATE CONSTRAINT article_id_unique IF NOT EXISTS
            FOR (a:Article)
            REQUIRE a.id IS UNIQUE
            """,
            database_=DATABASE
        )

        driver.execute_query(
            """
            CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
            FOR (c:Chunk)
            REQUIRE c.chunk_id IS UNIQUE
            """,
            database_=DATABASE
        )

        # 2. Nhap cac bai viet
        article_query = """
        UNWIND $rows AS row
        MERGE (a:Article {id: row.id})
        SET a.title = row.title,
            a.url = row.url,
            a.source = row.source,
            a.domain = row.domain,
            a.content_length = row.content_length,
            a.crawled_at = row.crawled_at
        """

        for batch in batches(articles, BATCH_SIZE):
            driver.execute_query(
                article_query,
                rows=batch,
                database_=DATABASE
            )

        print("Da nhap xong Article!")

        # 3. Nhap chunk va noi voi bai viet
        chunk_query = """
        UNWIND $rows AS row

        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c.article_id = row.article_id,
            c.title = row.title,
            c.url = row.url,
            c.source = row.source,
            c.chunk_index = row.chunk_index,
            c.content = row.content

        WITH row, c
        MATCH (a:Article {id: row.article_id})
        MERGE (a)-[:HAS_CHUNK]->(c)
        """

        for batch in batches(chunks, BATCH_SIZE):
            driver.execute_query(
                chunk_query,
                rows=batch,
                database_=DATABASE
            )

        print("Da nhap xong Chunk va HAS_CHUNK!")

        # 4. Kiem tra ket qua
        result, _, _ = driver.execute_query(
            """
            MATCH (a:Article)
            RETURN count(a) AS total
            """,
            database_=DATABASE
        )
        print("Tong Article:", result[0]["total"])

        result, _, _ = driver.execute_query(
            """
            MATCH (c:Chunk)
            RETURN count(c) AS total
            """,
            database_=DATABASE
        )
        print("Tong Chunk:", result[0]["total"])

        result, _, _ = driver.execute_query(
            """
            MATCH (:Article)-[r:HAS_CHUNK]->(:Chunk)
            RETURN count(r) AS total
            """,
            database_=DATABASE
        )
        print("Tong HAS_CHUNK:", result[0]["total"])

    finally:
        driver.close()


if __name__ == "__main__":
    main()
