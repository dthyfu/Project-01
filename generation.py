
import json
import re


def normalize_text(text):
    """Chuẩn hóa văn bản để phát hiện nội dung trùng lặp."""
    return " ".join(str(text or "").split()).casefold()


def prepare_sources(results):
    """
    Loại bỏ chunk rỗng, nội dung trùng lặp
    và gom các chunk theo URL bài viết.
    """
    sources = []
    seen_content = set()
    source_indices = {}

    for chunk in results:
        if not isinstance(chunk, dict):
            continue

        title = str(
            chunk.get("title") or "Tài liệu Long Châu"
        ).strip()

        url = str(chunk.get("url") or "").strip()
        content = str(chunk.get("content") or "").strip()

        if not content:
            continue

        normalized = normalize_text(content)

        if normalized in seen_content:
            continue

        seen_content.add(normalized)

        # Ưu tiên gom các đoạn cùng URL.
        key = ("url", url) if url else (
            "title", normalize_text(title)
        )

        if key in source_indices:
            index = source_indices[key]
            sources[index]["contents"].append(content)
        else:
            index = len(sources)

            sources.append({
                "title": title,
                "url": url,
                "contents": [content]
            })

            source_indices[key] = index

    return sources


def format_sources(sources):
    """Tạo danh sách nguồn tham khảo."""
    if not sources:
        return ""

    lines = ["\n\nNguồn tham khảo:"]

    for i, source in enumerate(sources, start=1):
        lines.append(f"[{i}] {source['title']}")

        if source["url"]:
            lines.append(source["url"])

    return "\n".join(lines)




def generate_answer(question, results):
    """
    Trích xuất câu trả lời ngắn gọn theo loại câu hỏi.
    Không sử dụng API hoặc mô hình ngôn ngữ.
    """

    if not results:
        return "Chưa tìm thấy tài liệu phù hợp để trả lời câu hỏi."

    question_lower = normalize_text(question)
    asks_for_food_recommendations = (
        any(term in question_lower for term in (
            "ăn gì", "nên ăn", "thực phẩm nào", "loại thực phẩm", "lựa chọn thực phẩm",
            "nên lựa chọn", "ăn những gì", "thực đơn", "bữa sáng",
        ))
        and not any(term in question_lower for term in ("hạn chế", "nên tránh", "không nên ăn", "kiêng"))
    )
    if asks_for_food_recommendations:
        return _grounded_food_recommendation(results, question)
    asks_type_comparison = (
        any(k in question_lower for k in ["khác nhau", "khác gì", "khác biệt", "phân biệt", "so sánh"])
        and "type 1" in question_lower and "type 2" in question_lower
    )
    if asks_type_comparison:
        return _grounded_type_comparison(results)
    asks_for_diabetes_types = (
        any(k in question_lower for k in ["loại nào", "những loại", "các loại", "mấy loại", "mấy tuýp", "các tuýp"])
        and any(k in question_lower for k in ["tiểu đường", "đái tháo đường", "diabetes", "tuýp", "type"])
    )
    if asks_for_diabetes_types:
        return _grounded_diabetes_types(results)

    # 1. Nhận diện loại câu hỏi và từ khóa phù hợp.
    if (
        any(k in question_lower for k in ["khác nhau", "khác gì", "khác biệt", "phân biệt", "so sánh", "thế nào"])
        and "type 1" in question_lower
        and "type 2" in question_lower
    ):
        intent = "type_comparison"
        keywords = [
            "type 1", "type 2", "tự miễn", "miễn dịch", "insulin",
            "kháng insulin", "tuyến tụy", "tế bào beta",
        ]

    elif any(k in question_lower for k in ["hạn chế ăn", "nên tránh", "không nên ăn", "kiêng ăn"]):
        intent = "nutrition_limit"
        keywords = ["hạn chế", "nên tránh", "không nên ăn", "kiêng", "đường", "bánh kẹo", "nước ngọt"]

    elif (
        any(k in question_lower for k in ["loại nào", "những loại", "các loại", "mấy loại", "mấy tuýp", "các tuýp"])
        and any(k in question_lower for k in ["tiểu đường", "đái tháo đường", "diabetes", "tuýp", "type"])
    ):
        intent = "types"
        keywords = [
            "type 1", "type 2", "tuýp 1", "tuýp 2", "đái tháo đường thai kỳ",
            "tiểu đường thai kỳ", "thai kỳ", "loại 1", "loại 2",
        ]

    elif any(k in question_lower for k in [
        "triệu chứng", "dấu hiệu", "biểu hiện"
    ]):
        intent = "symptom"
        keywords = [
            "khát nước", "hay khát", "khát nhiều",
            "đi tiểu nhiều", "tiểu nhiều", "tiểu thường xuyên",
            "mệt mỏi", "sụt cân", "giảm cân",
            "nhìn mờ", "mờ mắt", "đói nhiều",
            "vết thương lâu lành", "ngứa",
            "tê bì", "khô miệng"
        ]

    elif any(k in question_lower for k in [
        "nguyên nhân", "vì sao", "tại sao", "do đâu",
        "nguyên nhân gây"
    ]):
        intent = "cause"
        keywords = [
            "insulin", "kháng insulin",
            "tuyến tụy", "sản xuất insulin",
            "thiếu insulin", "thiếu hụt insulin",
            "rối loạn chuyển hóa", "yếu tố di truyền",
            "di truyền", "yếu tố nguy cơ",
            "thừa cân", "béo phì",
            "tế bào beta", "đường huyết"
        ]

    elif any(k in question_lower for k in [
        "biến chứng", "tác hại", "hậu quả",
        "ảnh hưởng", "nguy hiểm"
    ]):
        intent = "complication"
        keywords = [
            "biến chứng", "tổn thương mắt",
            "tổn thương thận", "suy thận",
            "hệ thần kinh", "tim mạch",
            "mạch máu", "võng mạc",
            "mù lòa", "đột quỵ",
            "bệnh tim", "loét bàn chân",
            "bàn chân đái tháo đường"
        ]

    elif any(k in question_lower for k in [
        "phòng ngừa", "phòng bệnh", "ngăn ngừa",
        "phòng tránh", "làm sao để tránh"
    ]):
        intent = "prevention"
        keywords = [
            "phòng ngừa", "phòng tránh",
            "kiểm soát cân nặng", "vận động",
            "tập thể dục", "chế độ ăn",
            "ăn uống lành mạnh", "hạn chế đường",
            "kiểm tra đường huyết", "lối sống"
        ]

    elif any(k in question_lower for k in [
        "điều trị", "chữa bệnh", "kiểm soát bệnh",
        "chữa như thế nào"
    ]):
        intent = "treatment"
        keywords = [
            "điều trị", "kiểm soát đường huyết",
            "insulin", "thuốc", "chế độ ăn",
            "vận động", "theo dõi đường huyết",
            "bác sĩ", "phác đồ điều trị"
        ]

    elif any(k in question_lower for k in [
        "là gì", "khái niệm", "định nghĩa",
        "là bệnh gì"
    ]):
        intent = "definition"
        keywords = [
            "là bệnh", "là một bệnh",
            "rối loạn chuyển hóa", "đường huyết",
            "insulin", "đường trong máu"
        ]

    else:
        intent = "general"
        keywords = [
            word for word in re.findall(r"\w+", question_lower)
            if len(word) > 2
        ]

    # 2. Từ khóa đặc trưng để tránh chọn câu chung chung.
    specific_keywords = {
        "type_comparison": [
            "type 1", "type 2", "tự miễn", "miễn dịch", "kháng insulin",
            "tuyến tụy", "tế bào beta",
        ],
        "nutrition_limit": ["hạn chế", "nên tránh", "không nên ăn", "kiêng", "đường", "bánh kẹo", "nước ngọt"],
        "types": [
            "type 1", "type 2", "tuýp 1", "tuýp 2", "đái tháo đường thai kỳ",
            "tiểu đường thai kỳ", "thai kỳ", "loại 1", "loại 2",
        ],
        "symptom": [
            "khát nước", "hay khát", "khát nhiều",
            "đi tiểu nhiều", "tiểu nhiều",
            "mệt mỏi", "sụt cân", "giảm cân",
            "nhìn mờ", "mờ mắt", "đói nhiều",
            "vết thương lâu lành", "ngứa", "tê bì"
        ],
        "cause": [
            "insulin", "kháng insulin", "tuyến tụy",
            "thiếu insulin", "di truyền", "béo phì",
            "thừa cân", "tế bào beta"
        ],
        "complication": [
            "tổn thương", "suy thận", "võng mạc",
            "mù lòa", "đột quỵ", "tim mạch",
            "hệ thần kinh", "loét bàn chân",
            "biến chứng"
        ],
        "prevention": [
            "phòng ngừa", "phòng tránh",
            "kiểm soát cân nặng", "tập thể dục",
            "vận động", "chế độ ăn"
        ],
        "treatment": [
            "điều trị", "insulin", "thuốc",
            "kiểm soát đường huyết", "bác sĩ"
        ],
        "definition": [
            "rối loạn chuyển hóa", "là bệnh",
            "là một bệnh", "đường huyết tăng cao"
        ]
    }

    # 3. Tách các chunk thành câu và chấm điểm.
    candidates = []
    seen = set()

    for chunk in results:
        if not isinstance(chunk, dict):
            continue

        content = str(chunk.get("content") or "").strip()
        if not content:
            continue

        sentences = re.split(r'(?<=[.!?])\s+|\n+', content)

        for sentence in sentences:
            sentence = sentence.strip(" \t\r\n-•")
            if len(sentence) < 25:
                continue

            normalized = normalize_text(sentence)

            if normalized in seen:
                continue
            seen.add(normalized)

            matched = [
                k for k in keywords if k in normalized
            ]
            score = len(matched)

            # Ưu tiên câu chứa nội dung đặc trưng của ý định.
            specific = specific_keywords.get(intent, [])
            specific_matches = [
                k for k in specific if k in normalized
            ]
            score += len(specific_matches) * 2

            if intent == "type_comparison" and not specific_matches:
                continue
            if intent == "types" and not specific_matches:
                continue
            if intent == "nutrition_limit" and not specific_matches:
                continue

            # Thưởng cho câu định nghĩa đúng trọng tâm.
            if intent == "definition" and any(
                phrase in normalized for phrase in [
                    "là một bệnh rối loạn chuyển hóa",
                    "là căn bệnh rối loạn chuyển hóa",
                    "là bệnh rối loạn chuyển hóa",
                    "đường huyết tăng cao"
                ]
            ):
                score += 4

            # Tránh chọn câu mở đầu chung chung khi hỏi triệu chứng.
            if intent == "symptom" and not specific_matches:
                continue

            # Với câu hỏi nguyên nhân, cần có nội dung giải thích
            # thay vì chỉ nhắc đến bệnh tiểu đường.
            if intent == "cause" and not specific_matches:
                continue

            if score > 0:
                candidates.append((score, sentence))

    # 4. Nếu không có câu phù hợp, không lấy bừa một đoạn.
    if not candidates:
        if intent in [
            "symptom", "cause", "complication",
            "prevention", "treatment", "definition", "nutrition_limit"
        ]:
            return (
                "Tài liệu truy xuất chưa cung cấp đủ thông tin "
                "để trả lời chính xác câu hỏi này."
            )

        return "Chưa tìm thấy nội dung phù hợp để trả lời."

    # 5. Xếp hạng và lấy các câu phù hợp nhất từ nhiều trang.
    candidates.sort(key=lambda item: item[0], reverse=True)

    answer_sentences = []
    total_length = 0

    for score, sentence in candidates:
        normalized = normalize_text(sentence)

        # Bỏ câu trùng hoặc chứa gần như toàn bộ câu đã chọn.
        if any(
            normalized in normalize_text(existing)
            or normalize_text(existing) in normalized
            for existing in answer_sentences
        ):
            continue

        if total_length + len(sentence) > 450:
            continue

        answer_sentences.append(sentence)
        total_length += len(sentence)

        max_sentences = 3 if intent in {"types", "type_comparison"} else 2
        if len(answer_sentences) >= max_sentences:
            break

    answer = " ".join(answer_sentences)

    # Giới hạn độ dài câu trả lời.
    words = answer.split()
    if len(words) > 80:
        answer = " ".join(words[:80]).rstrip(" ,;:") + "..."

    return answer


def _grounded_food_recommendation(results, question=""):
    """Return a short, evidence-derived answer for a broad food recommendation."""
    text = " ".join(
        str(item.get("content") or "") for item in results if isinstance(item, dict)
    ).casefold()
    if not any(marker in text for marker in (
        "nên ăn thực phẩm", "lựa chọn thực phẩm", "nhóm thực phẩm", "nên bổ sung",
    )):
        return "Tài liệu truy xuất chưa nêu đủ nhóm thực phẩm để trả lời chính xác."

    groups = (
        ("rau xanh", ("rau xanh", "rau")),
        ("trái cây", ("trái cây",)),
        ("ngũ cốc nguyên hạt", ("ngũ cốc nguyên hạt", "ngũ cốc nguyên")),
        ("cá", ("cá",)),
        ("đậu", ("đậu",)),
        ("thịt nạc", ("thịt nạc",)),
        ("sữa không béo", ("sữa không béo",)),
    )
    supported = [label for label, variants in groups if any(term in text for term in variants)]
    if len(supported) < 2:
        return "Tài liệu truy xuất chưa nêu đủ nhóm thực phẩm để trả lời chính xác."
    selected = supported[:3]
    if len(selected) > 1:
        food_list = ", ".join(selected[:-1]) + " và " + selected[-1]
    else:
        food_list = selected[0]
    answer = "Có thể ưu tiên " + food_list + "."
    if any(term in str(question).casefold() for term in ("thực đơn", "bữa sáng", "bữa trưa", "bữa tối")):
        answer += " Tài liệu truy xuất chưa đủ chi tiết để lập thực đơn mẫu ba bữa phù hợp cho từng cá nhân."
    return answer


def _grounded_diabetes_types(results):
    text = " ".join(
        str(item.get("content") or "") for item in results if isinstance(item, dict)
    ).casefold()
    categories = []
    if any(term in text for term in ("type 1", "tuýp 1", "tuyp 1", "loại 1")):
        categories.append("type 1")
    if any(term in text for term in ("type 2", "tuýp 2", "tuyp 2", "loại 2")):
        categories.append("type 2")
    if any(term in text for term in ("tiểu đường thai kỳ", "đái tháo đường thai kỳ", "thai kỳ")):
        categories.append("đái tháo đường thai kỳ")
    if len(categories) < 2:
        return "Tài liệu truy xuất chưa nêu đủ các loại bệnh tiểu đường để tổng hợp."
    return "Các dạng thường gặp được tài liệu nêu gồm " + ", ".join(categories) + "."


def _grounded_type_comparison(results):
    """Summarize only when the retrieved evidence supports both mechanisms."""
    text = " ".join(
        str(item.get("content") or "") for item in results if isinstance(item, dict)
    ).casefold()
    has_type_1_evidence = (
        any(term in text for term in ("type 1", "tuýp 1", "tự miễn", "miễn dịch"))
        and any(term in text for term in ("tự miễn", "miễn dịch", "sản xuất insulin", "tế bào beta", "thiếu insulin"))
    )
    has_type_2_evidence = (
        any(term in text for term in ("type 2", "tuýp 2"))
        and any(term in text for term in ("kháng insulin", "đề kháng insulin", "đề kháng với insulin"))
    )
    if not (has_type_1_evidence and has_type_2_evidence):
        return "Tài liệu truy xuất chưa nêu đủ điểm khác nhau giữa type 1 và type 2."
    return (
        "Type 1 thường do phản ứng tự miễn làm giảm khả năng sản xuất insulin; "
        "type 2 thường liên quan đến đề kháng insulin, khiến cơ thể sử dụng insulin kém hiệu quả."
    )



def generate_financial_answer(question, result):
    """Diễn đạt kết quả Stock/Gold thành câu tự nhiên."""

    from datetime import datetime

    if not isinstance(result, dict):
        return "Không thể xử lý kết quả truy vấn tài chính."

    rows = result.get("data", [])
    source = result.get("source", "Financial")

    if not rows:
        return "Chưa tìm thấy dữ liệu tài chính phù hợp."

    q = question.casefold()

    def format_date(value):
        try:
            return datetime.strptime(
                str(value)[:10], "%Y-%m-%d"
            ).strftime("%d/%m/%Y")
        except (ValueError, TypeError):
            return str(value or "")

    def format_number(value):
        try:
            return f"{float(value):,.2f}".rstrip("0").rstrip(".")
        except (ValueError, TypeError):
            return str(value)

    # Xác định tài sản được hỏi
    if any(k in q for k in ("google", "cổ phiếu", "chứng khoán", "stock")):
        asset = "cổ phiếu Google"
    elif any(k in q for k in ("vàng", "gold", "uso", "dầu")):
        asset = "vàng" if any(k in q for k in ("vàng", "gold")) else "tài sản USO"
    else:
        asset = "dữ liệu " + str(source)

    # Xác định loại giá được hỏi
    if "đóng cửa" in q or "close" in q:
        field = "close"
        price_type = "giá đóng cửa"
    elif "mở cửa" in q or "open" in q:
        field = "open"
        price_type = "giá mở cửa"
    elif "low" in q or "thấp" in q:
        field = "low"
        price_type = "giá thấp nhất"
    elif "high" in q or "cao" in q:
        field = "high"
        price_type = "giá cao nhất"
    else:
        field = "value"
        price_type = "giá trị"

    # Xác định yêu cầu tìm cực trị
    wants_max = any(k in q for k in ("cao nhất", "lớn nhất", "đỉnh"))
    wants_min = any(k in q for k in ("thấp nhất", "nhỏ nhất"))

    valid_rows = [
        row for row in rows
        if isinstance(row, dict) and row.get(field) is not None
    ]

    # Nếu có nhiều bản ghi, chọn giá trị lớn nhất/nhỏ nhất
    if valid_rows and (wants_max or wants_min):
        try:
            selector = max if wants_max else min
            rows = [selector(valid_rows, key=lambda r: float(r[field]))]
        except (ValueError, TypeError):
            pass

    row = rows[0]

    if not isinstance(row, dict):
        return f"Kết quả truy vấn {asset}: {row}."

    value = row.get(field)
    if value is None:
        value = row.get("value")

    if value is None:
        return (
            f"Đã tìm thấy dữ liệu về {asset}, "
            "nhưng chưa có trường giá phù hợp để trả lời."
        )

    # Nêu rõ cao nhất/thấp nhất nếu câu hỏi có yêu cầu
    if wants_max:
        description = (
            "giá đóng cửa cao nhất" if field == "close"
            else "giá mở cửa cao nhất" if field == "open"
            else "giá cao nhất"
        )
    elif wants_min:
        description = (
            "giá đóng cửa thấp nhất" if field == "close"
            else "giá mở cửa thấp nhất" if field == "open"
            else "giá thấp nhất"
        )
    else:
        description = price_type

    date = format_date(row.get("date", ""))
    value_text = format_number(value)

    # Đơn vị phải dựa trên schema dữ liệu thực tế.
    unit = " USD" if source == "Stock" else ""

    answer = (
        f"{description.capitalize()} của {asset} "
        f"trong dữ liệu là {value_text}{unit}"
    )

    if date:
        answer += f", ghi nhận vào ngày {date}"

    return answer + "."

