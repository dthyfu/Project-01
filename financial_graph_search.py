
from pathlib import Path
import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv(
    "NEO4J_USERNAME",
    os.getenv("NEO4J_USER", "neo4j")
)
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def run_query(query, parameters=None):
    """Thuc thi truy van Neo4j va tra ve danh sach dict."""
    if not PASSWORD:
        raise ValueError("Thieu NEO4J_PASSWORD trong .env")

    with GraphDatabase.driver(
        URI, auth=(USERNAME, PASSWORD)
    ) as driver:
        driver.verify_connectivity()

        records, _, _ = driver.execute_query(
            query,
            parameters_=parameters or {},
            database_=DATABASE
        )

        return [record.data() for record in records]


def extract_date(question):
    """Lay ngay YYYY-MM-DD neu cau hoi co ngay cu the."""
    match = re.search(
        r"\b(20\d{2}-\d{2}-\d{2})\b",
        question
    )
    return match.group(1) if match else None


def search_stock(question):
    """Truy van du lieu Google Stock."""
    q = question.lower()

    # Tra cuu theo ngay
    target_date = extract_date(question)
    if target_date:
        query = """
        MATCH (s:StockRecord)
        WHERE toString(s.date) = $target_date
        RETURN s.date AS date,
               s.open AS open,
               s.high AS high,
               s.low AS low,
               s.close AS close,
               s.volume AS volume
        LIMIT 1
        """
        return run_query(query, {"target_date": target_date})

    # Khoi luong giao dich lon nhat
    if (
        any(k in q for k in ["khối lượng", "khoi luong", "volume"])
        and any(k in q for k in [
            "cao nhất", "cao nhat", "lớn nhất", "lon nhat", "max"
        ])
    ):
        query = """
        MATCH (s:StockRecord)
        WHERE s.volume IS NOT NULL
        RETURN s.date AS date, s.volume AS volume
        ORDER BY s.volume DESC
        LIMIT 1
        """
        return run_query(query)

    # Gia dong cua trung binh
    if any(k in q for k in ["trung bình", "trung binh", "average", "avg"]):
        query = """
        MATCH (s:StockRecord)
        WHERE s.close IS NOT NULL
        RETURN avg(toFloat(s.close)) AS average_close
        """
        return run_query(query)

    # Phan biet gia cao nhat/thap nhat trong ngay
    # voi gia dong cua cao nhat/thap nhat
    intraday_high = (
        "trong ngày" in q or "trong ngay" in q
        or "high" in q
    )
    wants_high = any(k in q for k in [
        "cao nhất", "cao nhat", "lớn nhất", "lon nhat", "max"
    ])
    wants_low = any(k in q for k in [
        "thấp nhất", "thap nhat", "nhỏ nhất", "nho nhat", "min"
    ])
    asks_price = any(k in q for k in [
        "giá", "gia", "close", "đóng cửa", "dong cua"
    ])

    if asks_price and (wants_high or wants_low):
        field = "high" if intraday_high else "close"
        order = "DESC" if wants_high else "ASC"

        query = f"""
        MATCH (s:StockRecord)
        WHERE s.{field} IS NOT NULL
        RETURN s.date AS date,
               s.{field} AS value,
               s.close AS close,
               s.high AS high,
               s.low AS low
        ORDER BY s.{field} {order}
        LIMIT 1
        """
        return run_query(query)

    # Mac dinh: 5 ban ghi gan nhat
    query = """
    MATCH (s:StockRecord)
    RETURN s.date AS date,
           s.open AS open,
           s.high AS high,
           s.low AS low,
           s.close AS close,
           s.volume AS volume
    ORDER BY s.date DESC
    LIMIT 5
    """
    return run_query(query)


def search_gold(question):
    """Truy van du lieu Gold trong Neo4j."""
    q = question.lower()

    # Tra cuu theo ngay
    target_date = extract_date(question)
    if target_date:
        query = """
        MATCH (g:GoldRecord)
        WHERE toString(g.date) = $target_date
        RETURN g.date AS date,
               g.Open AS open,
               g.High AS high,
               g.Low AS low,
               g.Close AS close
        LIMIT 1
        """
        return run_query(query, {"target_date": target_date})

    # Gia dong cua trung binh
    if any(k in q for k in ["trung bình", "trung binh", "average", "avg"]):
        query = """
        MATCH (g:GoldRecord)
        WHERE g.Close IS NOT NULL
        RETURN avg(toFloat(g.Close)) AS average_close
        """
        return run_query(query)

    # Phan biet gia cao nhat/thap nhat trong ngay
    # voi gia dong cua cao nhat/thap nhat
    intraday_high = (
        "trong ngày" in q or "trong ngay" in q
        or "high" in q
    )
    wants_high = any(k in q for k in [
        "cao nhất", "cao nhat", "lớn nhất", "lon nhat", "max"
    ])
    wants_low = any(k in q for k in [
        "thấp nhất", "thap nhat", "nhỏ nhất", "nho nhat", "min"
    ])
    asks_price = any(k in q for k in [
        "giá", "gia", "close", "đóng cửa", "dong cua"
    ])

    if asks_price and (wants_high or wants_low):
        field = "High" if intraday_high else "Close"
        order = "DESC" if wants_high else "ASC"

        query = f"""
        MATCH (g:GoldRecord)
        WHERE g.{field} IS NOT NULL
        RETURN g.date AS date,
               g.{field} AS value,
               g.Close AS close,
               g.High AS high,
               g.Low AS low
        ORDER BY g.{field} {order}
        LIMIT 1
        """
        return run_query(query)

    # Mac dinh: 5 ban ghi gan nhat
    query = """
    MATCH (g:GoldRecord)
    RETURN g.date AS date,
           g.Close AS close,
           g.High AS high,
           g.Low AS low,
           g.GDX_Close AS gdx_close,
           g.USO_Close AS uso_close
    ORDER BY g.date DESC
    LIMIT 5
    """
    return run_query(query)


def search_financial(question):
    """
    Chon Stock hoac Gold dua tren tu khoa.
    Tra ve du lieu Neo4j, chua phai cau tra loi tu LLM.
    """
    q = question.lower()

    stock_keywords = [
        "google", "cổ phiếu", "co phieu",
        "stock", "volume google"
    ]
    gold_keywords = [
        "gold", "vàng", "vang", "gdx", "uso", "giá vàng", "gia vang"
    ]

    if any(k in q for k in stock_keywords):
        return {
            "source": "Stock",
            "data": search_stock(question)
        }

    if any(k in q for k in gold_keywords):
        return {
            "source": "Gold",
            "data": search_gold(question)
        }

    return {
        "source": "unknown",
        "data": [],
        "message": "Chua xac dinh duoc cau hoi thuoc Stock hay Gold."
    }


if __name__ == "__main__":
    question = input("Nhap cau hoi ve Stock hoac Gold: ")
    result = search_financial(question)

    print("\nNguon:", result["source"])

    for item in result["data"]:
        print(item)

    if not result["data"]:
        print(result.get("message", "Khong tim thay du lieu."))
