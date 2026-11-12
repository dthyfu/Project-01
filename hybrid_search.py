
import json
import logging
import os
import re
from time import perf_counter
import unicodedata
from pathlib import Path
from generation import (
    generate_answer,
)

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ==================== CAU HINH ====================

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "data" / "longchau_chunks.json"
VECTOR_FILE = BASE_DIR / "data" / "longchau_vectors.npy"
ENTITY_FILE = BASE_DIR / "data" / "longchau_entities_normalized.json"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5
GRAPH_TOP_K = 1000
RRF_K = 60

load_dotenv(BASE_DIR / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# ==================== TIEN ICH ====================

def normalize_text(text):
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


def reciprocal_rank_fusion(*rankings, k=RRF_K):
    """Hop nhat cac danh sach xep hang bang RRF."""
    fused_scores = {}

    for ranking in rankings:
        for rank, chunk_index in enumerate(ranking, start=1):
            fused_scores[chunk_index] = (
                fused_scores.get(chunk_index, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        fused_scores,
        key=lambda index: fused_scores[index],
        reverse=True,
    ), fused_scores


def get_question_intent(question):
    """Xác định ý định chính của câu hỏi y tế."""
    question = normalize_text(question)

    if (
        any(word in question for word in ("khác nhau", "khác gì", "phân biệt", "so sánh", "thế nào"))
        and any(word in question for word in ("type 1", "tuýp 1", "type 2", "tuýp 2"))
    ):
        return "type_comparison"

    if any(word in question for word in (
        "triệu chứng", "dấu hiệu", "biểu hiện"
    )):
        return "symptom"

    if any(word in question for word in (
        "biến chứng", "hậu quả"
    )):
        return "complication"

    if any(word in question for word in (
        "nguyên nhân", "tại sao", "vì sao"
    )):
        return "cause"

    if any(word in question for word in (
        "phòng ngừa", "phòng bệnh", "ngăn ngừa"
    )):
        return "prevention"

    if any(word in question for word in (
        "điều trị", "chữa bệnh", "điều trị như thế nào"
    )):
        return "treatment"

    if any(word in question for word in (
        "là gì", "khái niệm", "định nghĩa", "thế nào là"
    )):
        return "definition"

    return "general"



def get_intent_bonus(content, intent, title=""):
    """Chấm điểm bổ sung theo ý định câu hỏi."""
    text = normalize_text(title + " " + content)

    if intent == "type_comparison":
        has_type_1 = any(label in text for label in ("type 1", "tuýp 1", "tup 1"))
        has_type_2 = any(label in text for label in ("type 2", "tuýp 2", "tup 2"))
        score = 0.04 if has_type_1 and has_type_2 else 0.012 if has_type_1 or has_type_2 else 0.0
        if any(term in text for term in ("tự miễn", "miễn dịch", "kháng insulin", "đề kháng insulin")):
            score += 0.01
        return score

    keywords = {
        "definition": (
            "là gì",
            "là bệnh",
            "là một bệnh",
            "là bệnh lý",
            "là tình trạng",
            "rối loạn chuyển hóa",
            "được đặc trưng bởi",
            "đường huyết tăng cao",
            "insulin",
        ),
        "symptom": (
            "triệu chứng",
            "dấu hiệu",
            "biểu hiện",
            "hay khát",
            "đi tiểu nhiều",
            "mệt mỏi",
        ),
        "complication": (
            "biến chứng",
            "tổn thương",
            "nguy cơ biến chứng",
        ),
        "cause": (
            "nguyên nhân",
            "do cơ thể",
            "dẫn đến",
            "gây ra",
            "kháng insulin",
        ),
        "prevention": (
            "phòng ngừa",
            "ngăn ngừa",
            "giảm nguy cơ",
            "lối sống lành mạnh",
        ),
        "treatment": (
            "điều trị",
            "phương pháp điều trị",
            "kiểm soát bệnh",
            "phác đồ",
        ),
    }

    if intent not in keywords:
        return 0.0

    matches = sum(
        1 for keyword in keywords[intent]
        if keyword in text
    )

    # Điểm cộng đủ lớn để tác động đến thứ hạng RRF.
    return min(matches, 4) * 0.005


def get_title_bonus(title, question):
    """Ưu tiên bài viết có tiêu đề khớp trực tiếp với câu hỏi."""
    title = normalize_text(title)
    question = normalize_text(question)

    if not title or not question:
        return 0.0

    # Bỏ dấu câu để so sánh tiêu đề và câu hỏi.
    clean_title = " ".join(tokenize(title))
    clean_question = " ".join(tokenize(question))

    if clean_title == clean_question:
        return 0.05

    # Ưu tiên tiêu đề chứa nguyên câu hỏi.
    if clean_question in clean_title:
        return 0.03

    return 0.0

# ==================== GRAPH SEARCH ====================

ALIASES = {
    "bệnh tiểu đường": "đái tháo đường",
    "tiểu đường type 1": "đái tháo đường type 1",
    "tiểu đường tuýp 1": "đái tháo đường type 1",
    "tiểu đường type 2": "đái tháo đường type 2",
    "tiểu đường tuýp 2": "đái tháo đường type 2",
}


def load_entity_dictionary():
    with open(ENTITY_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    unique_entities = {}

    for item in data:
        for entity in item.get("entities", []):
            key = (entity["name"], entity["type"])
            unique_entities[key] = {
                "name": entity["name"],
                "type": entity["type"],
            }

    return list(unique_entities.values())


def find_query_entities(question, dictionary):
    normalized_question = normalize_text(question)

    for alias, canonical in ALIASES.items():
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        normalized_question = re.sub(
            pattern, canonical, normalized_question
        )

    found = []

    for entity in dictionary:
        name = normalize_text(entity["name"])
        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"

        if re.search(pattern, normalized_question):
            found.append(entity)

    return found


def get_target_type(question):
    question = normalize_text(question)

    if any(word in question for word in (
        "triệu chứng", "dấu hiệu", "biểu hiện"
    )):
        return "Symptom"

    if any(word in question for word in (
        "biến chứng", "hậu quả"
    )):
        return "Complication"

    if any(word in question for word in (
        "yếu tố nguy cơ", "nguy cơ mắc"
    )):
        return "RiskFactor"

    return ""


def graph_search(driver, question, dictionary):
    entities = find_query_entities(question, dictionary)

    if not entities:
        print(
            "Graph Search: khong tim thay thuc the khop "
            "trong cau hoi."
        )
        return []

    print("\nThuc the Graph Search:")
    for entity in entities:
        print(f"- [{entity['type']}] {entity['name']}")

    query = """
    UNWIND $entities AS item

    MATCH (e:Entity {name: item.name, type: item.type})
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
         END) AS related_entities

    WITH c, a, matched_entities, matched_count,
         [x IN related_entities WHERE x IS NOT NULL]
             AS target_entities

    RETURN
        c.chunk_id AS chunk_id,
        matched_entities,
        matched_count,
        target_entities,
        size(target_entities) AS target_count

    ORDER BY target_count DESC, matched_count DESC, chunk_id ASC
    LIMIT $top_k
    """

    rows = [
        {"name": entity["name"], "type": entity["type"]}
        for entity in entities
    ]

    records, _, _ = driver.execute_query(
        query,
        entities=rows,
        target_type=get_target_type(question),
        top_k=GRAPH_TOP_K,
        database_=NEO4J_DATABASE,
    )

    return [record.data() for record in records]

# ==================== REUSABLE HYBRID RAG ====================

class HybridRAG:
    """Khoi tao cac thanh phan mot lan, tai su dung cho nhieu cau hoi."""

    def __init__(self):
        for path in [CHUNKS_FILE, VECTOR_FILE, ENTITY_FILE]:
            if not path.exists():
                raise FileNotFoundError(f"Khong tim thay file: {path}")

        if not all((NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)):
            raise ValueError("Thieu cau hinh Neo4j trong file .env.")

        with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        self.embeddings = np.load(VECTOR_FILE)

        if len(self.chunks) != len(self.embeddings):
            raise ValueError("So chunks khong khop so vector.")

        self.chunk_id_to_index = {
            str(chunk.get("chunk_id")): index
            for index, chunk in enumerate(self.chunks)
        }

        print("Dang khoi tao BM25...")
        self.bm25 = BM25Okapi([
            tokenize(chunk.get("content", ""))
            for chunk in self.chunks
        ])

        print("Dang tai mo hinh embedding...")
        self.model = SentenceTransformer(MODEL_NAME)

        self.dictionary = load_entity_dictionary()

        print("Dang ket noi Neo4j...")
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        self.driver.verify_connectivity()

        print("Hybrid RAG da san sang!")

    
    def search(self, question, top_k=TOP_K):
        question = question.strip()

        if not question:
            return []

        search_started = perf_counter()
        stage_ms = {}
        intent = get_question_intent(question)
        # Số ứng viên lấy từ mỗi phương pháp trước khi hợp nhất.
        candidate_k = max(30, top_k * 10)

        # 1. BM25: chỉ lấy những chunk đứng đầu.
        stage_started = perf_counter()
        bm25_scores = self.bm25.get_scores(tokenize(question))

        bm25_ranking = np.argsort(
            bm25_scores
        )[::-1][:candidate_k].tolist()
        stage_ms["bm25"] = round((perf_counter() - stage_started) * 1000)

        # 2. Vector Search.
        stage_started = perf_counter()
        query_vector = self.model.encode(
            [question],
            normalize_embeddings=True,
        )[0]

        if self.embeddings.ndim != 2 or query_vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: query={query_vector.shape[0]}, "
                f"index={self.embeddings.shape if self.embeddings.ndim == 2 else self.embeddings.ndim}."
            )

        vector_scores = np.dot(self.embeddings, query_vector)

        vector_ranking = np.argsort(
            vector_scores
        )[::-1][:candidate_k].tolist()
        stage_ms["vector"] = round((perf_counter() - stage_started) * 1000)

        # 3. Graph Search.
        stage_started = perf_counter()
        graph_results = graph_search(
            self.driver,
            question,
            self.dictionary,
        )

        graph_ranking = []
        seen_graph = set()

        for result in graph_results:
            index = self.chunk_id_to_index.get(
                str(result.get("chunk_id"))
            )

            if index is not None and index not in seen_graph:
                graph_ranking.append(index)
                seen_graph.add(index)
        stage_ms["graph"] = round((perf_counter() - stage_started) * 1000)

        # 4. Hợp nhất thứ hạng bằng RRF.
        stage_started = perf_counter()
        hybrid_ranking, hybrid_scores = reciprocal_rank_fusion(
            bm25_ranking,
            vector_ranking,
            graph_ranking,
        )
        stage_ms["rrf"] = round((perf_counter() - stage_started) * 1000)

        
        # 5. Xếp hạng lại theo ý định câu hỏi.
        stage_started = perf_counter()
        reranked = []

        for index in hybrid_ranking:
            chunk = self.chunks[index]

            content = chunk.get("content", "")
            title = chunk.get("title", "")

            
            base_score = hybrid_scores[index]

            intent_score = get_intent_bonus(
                content,
                intent,
                title,
            )

            title_score = get_title_bonus(
                title,
                question,
            )

            final_score = (
                base_score
                + intent_score
                + title_score
            )


            reranked.append((
                index,
                final_score,
                base_score,
                intent_score,
            ))

        reranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # 6. Loại bỏ chunk trùng nội dung.
        results = []
        seen_contents = set()

        for index, final_score, base_score, intent_score in reranked:
            chunk = self.chunks[index]
            content = chunk.get("content", "").strip()

            normalized_content = normalize_text(content)

            if not normalized_content:
                continue

            if normalized_content in seen_contents:
                continue

            seen_contents.add(normalized_content)

            result = dict(chunk)
            result["rrf_score"] = round(
                float(base_score), 6
            )
            result["intent_score"] = round(
                float(intent_score), 6
            )
            result["final_score"] = round(
                float(final_score), 6
            )

            results.append(result)

            if len(results) >= top_k:
                break

        stage_ms["rerank_and_select"] = round((perf_counter() - stage_started) * 1000)
        stage_ms["total"] = round((perf_counter() - search_started) * 1000)
        self.last_search_diagnostics = {
            "intent": intent,
            "candidate_k": candidate_k,
            "bm25_top": [str(self.chunks[index].get("chunk_id", index)) for index in bm25_ranking[:5]],
            "vector_top": [str(self.chunks[index].get("chunk_id", index)) for index in vector_ranking[:5]],
            "graph_top": [str(self.chunks[index].get("chunk_id", index)) for index in graph_ranking[:5]],
            "rrf_top": [str(self.chunks[index].get("chunk_id", index)) for index in hybrid_ranking[:5]],
            "selected": [str(item.get("chunk_id", "")) for item in results],
            "timings_ms": stage_ms,
        }
        logger.info(
            "hybridrag search intent=%s candidate_k=%d bm25_top=%s vector_top=%s graph_top=%s rrf_top=%s selected=%s timings_ms=%s",
            intent, candidate_k, self.last_search_diagnostics["bm25_top"],
            self.last_search_diagnostics["vector_top"], self.last_search_diagnostics["graph_top"],
            self.last_search_diagnostics["rrf_top"], self.last_search_diagnostics["selected"], stage_ms,
        )
        return results


    
    def ask(self, question, domain="diabetes"):
        question = question.strip()
        domain = domain.strip().lower()

        if not question:
            return {
                "answer": "Bạn chưa nhập câu hỏi.",
                "sources": [],
                "retrieved_count": 0,
            }

        if domain != "diabetes":
            return {
                "answer": "Lĩnh vực không hợp lệ.",
                "sources": [],
                "retrieved_count": 0,
            }

        # Chỉ chạy pipeline tài liệu về bệnh tiểu đường.
        results = self.search(question)

        if not results:
            return {
                "answer": "Chưa tìm thấy tài liệu phù hợp.",
                "sources": [],
                "retrieved_count": 0,
            }

        answer = generate_answer(question, results)

        # 3. Thu thập nguồn Diabetes và loại URL trùng lặp
        sources = []
        seen_urls = set()

        for chunk in results:
            url = chunk.get("url", "")
            key = url or chunk.get("title", "")

            if key in seen_urls:
                continue

            seen_urls.add(key)

            sources.append({
                "title": chunk.get("title", "Tài liệu Long Châu"),
                "url": url,
            })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(results),
        }



    def close(self):
        self.driver.close()


# ==================== HYBRID SEARCH ====================

def main():
    rag = None

    try:
        # Khoi tao BM25, embedding model va Neo4j mot lan
        rag = HybridRAG()

        print("\nHybrid RAG san sang.")
        print("Nhap 'exit' de thoat.")

        while True:
            question = input(
                "\nNhap cau hoi: "
            ).strip()

            if question.lower() in ("exit", "quit"):
                break

            if not question:
                continue

            try:
                result = rag.ask(question)

                print("\n========== CAU TRA LOI ==========")
                print(result["answer"])

            except Exception as error:
                print(
                    "\nLoi khi xu ly cau hoi:",
                    str(error)
                )

    except Exception as error:
        print("\nKhong the khoi tao Hybrid RAG:", str(error))

    finally:
        if rag is not None:
            rag.close()

        print("\nDa dong Hybrid RAG.")


if __name__ == "__main__":
    main()

