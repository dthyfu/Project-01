
import json
from collections import defaultdict

import numpy as np
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from hybrid_search import (
    BASE_DIR,
    CHUNKS_FILE,
    VECTOR_FILE,
    MODEL_NAME,
    GRAPH_TOP_K,
    RRF_K,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    tokenize,
    reciprocal_rank_fusion,
    load_entity_dictionary,
    graph_search,
)


# ==================================================
# BO CAU HOI DANH GIA
# Mỗi nhóm từ khóa dùng để xác định bài viết liên quan
# trong corpus hiện có.
# ==================================================

TEST_CASES = [
    {
        "question": "Bệnh tiểu đường có những triệu chứng nào?",
        "relevant_titles": [
            "triệu chứng tiểu đường",
            "dấu hiệu buổi sáng",
            "bệnh tiểu đường có mấy tuýp",
        ],
    },
    {
        "question": "Những dấu hiệu sớm của bệnh tiểu đường là gì?",
        "relevant_titles": [
            "triệu chứng giai đoạn đầu",
            "dấu hiệu buổi sáng",
        ],
    },
    {
        "question": "Bệnh tiểu đường có những loại nào?",
        "relevant_titles": [
            "bệnh tiểu đường có mấy tuýp",
            "đái tháo đường type",
        ],
    },
    {
        "question": "Người mắc tiểu đường nên ăn gì?",
        "relevant_titles": [
            "ăn gì để tăng lượng insulin",
            "thực phẩm",
            "chế độ ăn",
        ],
    },
    {
        "question": "Insulin có vai trò gì trong bệnh tiểu đường?",
        "relevant_titles": [
            "insulin",
        ],
    },
    {
        "question": "Bệnh tiểu đường có thể gây ra biến chứng gì?",
        "relevant_titles": [
            "biến chứng",
        ],
    },
    {
        "question": "Làm thế nào để phòng ngừa bệnh tiểu đường?",
        "relevant_titles": [
            "phòng ngừa bệnh tiểu đường",
            "phòng bệnh tiểu đường",
        ],
    },
    {
        "question": "Đường huyết cao có liên quan gì đến bệnh tiểu đường?",
        "relevant_titles": [
            "đường huyết",
            "glucose máu",
        ],
    },
]

K = 5


def normalize_title(text):
    return " ".join((text or "").lower().split())


def get_article_key(chunk):
    """Gộp các chunks thuộc cùng một bài viết."""
    article_id = chunk.get("article_id")
    if article_id is not None:
        return str(article_id)

    return chunk.get("title", "").strip()


def retrieve_articles(question, chunks, embeddings, bm25,
                      model, driver, dictionary):
    """Chạy ba nhánh tìm kiếm và trả về thứ hạng bài viết."""

    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_ranking = np.argsort(bm25_scores)[::-1].tolist()

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
    )[0]

    vector_scores = np.dot(embeddings, query_vector)
    vector_ranking = np.argsort(vector_scores)[::-1].tolist()

    graph_results = graph_search(driver, question, dictionary)

    chunk_id_to_index = {
        str(chunk.get("chunk_id")): i
        for i, chunk in enumerate(chunks)
    }

    graph_ranking = []
    seen_graph_indices = set()

    for result in graph_results:
        index = chunk_id_to_index.get(str(result["chunk_id"]))

        if index is not None and index not in seen_graph_indices:
            graph_ranking.append(index)
            seen_graph_indices.add(index)

    hybrid_ranking, _ = reciprocal_rank_fusion(
        bm25_ranking,
        vector_ranking,
        graph_ranking,
        k=RRF_K,
    )

    # Lấy tối đa K bài viết khác nhau, không đếm trùng chunks.
    ranked_articles = []
    seen_articles = set()

    for index in hybrid_ranking:
        chunk = chunks[index]
        article_key = get_article_key(chunk)

        if article_key and article_key not in seen_articles:
            seen_articles.add(article_key)
            ranked_articles.append({
                "key": article_key,
                "title": chunk.get("title", ""),
            })

        if len(ranked_articles) >= K:
            break

    return ranked_articles


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(VECTOR_FILE)

    if len(chunks) != len(embeddings):
        raise ValueError("Số chunks không khớp số vectors.")

    print(f"Tổng số chunks: {len(chunks)}")
    print("Đang chuẩn bị BM25...")

    bm25 = BM25Okapi([
        tokenize(chunk.get("content", ""))
        for chunk in chunks
    ])

    print("Đang tải mô hình embedding...")
    model = SentenceTransformer(MODEL_NAME)

    dictionary = load_entity_dictionary()

    if not all((NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)):
        raise ValueError("Thiếu cấu hình Neo4j trong file .env.")

    # Xác định các bài viết liên quan theo tiêu đề.
    # Đây là nhãn tham chiếu ban đầu, cần được kiểm tra khi báo cáo.
    titles = [
        normalize_title(chunk.get("title", ""))
        for chunk in chunks
    ]

    gold_sets = []

    for case in TEST_CASES:
        matched = set()

        for chunk in chunks:
            title = normalize_title(chunk.get("title", ""))

            if any(
                keyword in title
                for keyword in case["relevant_titles"]
            ):
                key = get_article_key(chunk)
                if key:
                    matched.add(key)

        gold_sets.append(matched)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    evaluated = 0
    skipped = 0
    total_hits = 0
    total_recall = 0.0
    total_rr = 0.0

    try:
        driver.verify_connectivity()

        print("\n========== EVALUATION ==========")

        for case, relevant_articles in zip(TEST_CASES, gold_sets):
            question = case["question"]

            if not relevant_articles:
                print(f"\n[BỎ QUA] Không tìm thấy nhãn tham chiếu:")
                print(question)
                skipped += 1
                continue

            print(f"\nCâu hỏi: {question}")

            ranked = retrieve_articles(
                question,
                chunks,
                embeddings,
                bm25,
                model,
                driver,
                dictionary,
            )

            retrieved_keys = [item["key"] for item in ranked]
            retrieved_set = set(retrieved_keys)

            relevant_found = retrieved_set & relevant_articles
            hit = 1 if relevant_found else 0

            recall = len(relevant_found) / len(relevant_articles)

            reciprocal_rank = 0.0
            for rank, key in enumerate(retrieved_keys, start=1):
                if key in relevant_articles:
                    reciprocal_rank = 1.0 / rank
                    break

            evaluated += 1
            total_hits += hit
            total_recall += recall
            total_rr += reciprocal_rank

            print("Bài viết trong Top 5:")
            for rank, item in enumerate(ranked, start=1):
                marker = (
                    " [LIÊN QUAN]"
                    if item["key"] in relevant_articles
                    else ""
                )
                print(f"  {rank}. {item['title']}{marker}")

            print(
                f"Hit@5={hit} | "
                f"Recall@5={recall:.3f} | "
                f"RR@5={reciprocal_rank:.3f}"
            )

    finally:
        driver.close()

    print("\n========== TONG KET ==========")
    print(f"Số câu hỏi đánh giá: {evaluated}")
    print(f"Số câu bị bỏ qua: {skipped}")

    if evaluated:
        print(f"Hit Rate@5: {total_hits / evaluated:.4f}")
        print(f"Recall@5:   {total_recall / evaluated:.4f}")
        print(f"MRR@5:      {total_rr / evaluated:.4f}")

        report = {
            "evaluated_questions": evaluated,
            "skipped_questions": skipped,
            "hit_rate_at_5": round(total_hits / evaluated, 4),
            "recall_at_5": round(total_recall / evaluated, 4),
            "mrr_at_5": round(total_rr / evaluated, 4),
            "note": (
                "Nhãn liên quan được tạo từ từ khóa tiêu đề; "
                "cần rà soát trước khi dùng làm kết quả khoa học."
            ),
        }

        output_path = BASE_DIR / "data" / "evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        print(f"\nĐã lưu kết quả: {output_path}")


if __name__ == "__main__":
    main()
