import io
import re
import math
import unicodedata
from contextlib import redirect_stdout

from schemas.messages import RetrievedChunk


class RetrievalService:
    """Adapter over the existing HybridRAG.search implementation."""

    def __init__(self, rag):
        self.rag = rag

    def search(self, question, top_k=5):
        # Existing retrieval prints Vietnamese debug lines; capture them so a
        # Windows console code page cannot turn a successful search into an error.
        with redirect_stdout(io.StringIO()):
            chunks = self.rag.search(question, top_k=top_k)
        context = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", ""))
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            context.append(RetrievedChunk(
                source_id=f"chunk:{chunk_id}" if chunk_id else "chunk:unknown",
                chunk_id=chunk_id,
                title=str(chunk.get("title") or ""),
                url=str(chunk.get("url") or ""),
                content=content,
                score=_as_float(chunk.get("final_score")),
            ))
        # Prefer distinct articles first so a synthesis can combine evidence
        # across pages instead of spending its context budget on one article.
        primary = []
        deferred = []
        for item in context:
            source = (item.url or item.title or item.source_id).strip().rstrip("/").casefold()
            (primary if not any(
                (selected.url or selected.title or selected.source_id).strip().rstrip("/").casefold() == source
                for selected in primary
            ) else deferred).append(item)
        diversified = primary[:top_k]
        counts = {}
        for item in diversified:
            key = (item.url or item.title or item.source_id).strip().rstrip("/").casefold()
            counts[key] = counts.get(key, 0) + 1
        for item in deferred:
            if len(diversified) >= top_k:
                break
            key = (item.url or item.title or item.source_id).strip().rstrip("/").casefold()
            if counts.get(key, 0) < 2:
                diversified.append(item)
                counts[key] = counts.get(key, 0) + 1
        return diversified


def _as_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def context_supports_question(question, context):
    """Conservative lexical evidence gate; it does not alter retrieval rank."""
    if not context:
        return False
    normalized_question = _normalize(question)
    document_text = " ".join(_normalize(item.title + " " + item.content) for item in context)
    asks_type_comparison = (
        any(term in normalized_question for term in ("khac nhau", "khac gi", "khac biet", "phan biet", "so sanh", "the nao"))
        and any(term in normalized_question for term in ("type 1", "tuyp 1", "type 2", "tuyp 2"))
    )
    if asks_type_comparison:
        text = document_text
        has_type_1 = any(term in text for term in ("type 1", "tuyp 1", "tup 1"))
        has_type_2 = any(term in text for term in ("type 2", "tuyp 2", "tup 2"))
        has_type_1_detail = any(term in text for term in (
            "tu mien", "mien dich", "te bao beta", "san xuat insulin", "thieu insulin",
        ))
        has_type_2_detail = any(term in text for term in (
            "khang insulin", "de khang insulin", "de khang cua co the", "insulin resistance",
        ))
        return has_type_1 and has_type_2 and has_type_1_detail and has_type_2_detail

    asks_food_recommendation = any(term in normalized_question for term in (
        "an gi", "nen an", "thuc pham nao", "loai thuc pham", "thuc don", "bua sang", "bua trua", "bua toi",
    ))
    if asks_food_recommendation:
        has_dietary_evidence = any(term in document_text for term in (
            "nen an", "lua chon thuc pham", "nhom thuc pham", "che do an", "dinh duong",
            "thuc pham giau chat xo", "rau xanh", "ngu coc nguyen hat",
        ))
        asks_type_2 = any(term in normalized_question for term in ("type 2", "tuyp 2", "tup 2"))
        type_2_evidence = any(term in document_text for term in ("type 2", "tuyp 2", "tup 2"))
        return has_dietary_evidence and (not asks_type_2 or type_2_evidence)

    all_query_terms = _terms(question)
    query_terms = all_query_terms - _STOP_WORDS
    document_terms = set()
    for item in context:
        document_terms.update(_terms(item.title))
        document_terms.update(_terms(item.content))
    if any(_phrase(phrase) not in document_text for phrase in _capitalized_phrases(question)):
        return False
    if not query_terms:
        return bool(all_query_terms & document_terms)
    overlap = query_terms & document_terms
    meaningful = query_terms - _DOMAIN_TERMS - _QUERY_NOISE
    if len(meaningful) >= 3:
        required = max(2, math.ceil(len(meaningful) * 0.6))
        return len(overlap & meaningful) >= required
    return bool(overlap & meaningful) if meaningful else bool(all_query_terms & document_terms)


def filter_relevant_chunks(question, context):
    """Drop obvious cross-topic chunks for intents with clear evidence markers."""
    normalized_question = _normalize(question)
    if not context:
        return []

    asks_comparison = (
        any(term in normalized_question for term in ("khac nhau", "khac gi", "phan biet", "so sanh", "the nao"))
        and any(term in normalized_question for term in ("type 1", "tuyp 1", "type 2", "tuyp 2"))
    )
    asks_types = (
        any(term in normalized_question for term in ("loai nao", "may loai", "nhung loai", "cac loai"))
        and any(term in normalized_question for term in ("tieu duong", "dai thao duong", "diabetes"))
    )
    asks_nutrition = any(term in normalized_question for term in (
        "dinh duong", "che do an", "thuc pham", "nen an", "an gi", "han che an", "khau phan",
    ))
    asks_hba1c = "hba1c" in normalized_question or "hemoglobin a1c" in normalized_question

    if not (asks_comparison or asks_types or asks_nutrition or asks_hba1c):
        return list(context)

    relevant = []
    for item in context:
        text = _normalize(item.title + " " + item.content)
        if asks_comparison:
            type_1 = any(term in text for term in ("type 1", "tuyp 1", "tup 1"))
            type_2 = any(term in text for term in ("type 2", "tuyp 2", "tup 2"))
            type_1_detail = any(term in text for term in (
                "tu mien", "mien dich", "te bao beta", "tuyen tuy", "san xuat insulin",
                "khong tao ra insulin", "khong tao insulin", "thieu insulin",
            ))
            type_2_detail = any(term in text for term in (
                "khang insulin", "de khang insulin", "de khang voi insulin", "insulin resistance",
                "su dung insulin khong hieu qua", "khong su dung insulin hieu qua",
            ))
            if (type_1 and type_2 and (type_1_detail or type_2_detail)) or (type_1 and type_1_detail) or (type_2 and type_2_detail):
                relevant.append(item)
        elif asks_types:
            labels = (
                ("type 1", "tuyp 1", "tup 1", "loai 1"),
                ("type 2", "tuyp 2", "tup 2", "loai 2"),
                ("thai ky", "thai nghen", "thai phu"),
            )
            if any(any(label in text for label in group) for group in labels):
                relevant.append(item)
        elif asks_nutrition:
            if any(term in text for term in (
                "dinh duong", "che do an", "thuc pham", "bua an", "khau phan", "an uong", "nen an", "han che an",
            )):
                relevant.append(item)
        elif asks_hba1c:
            if "hba1c" in text or "hemoglobin a1c" in text:
                relevant.append(item)

    return relevant


def answer_supported_by_context(answer, context):
    """Reject answers with too little lexical evidence in the retrieved chunks."""
    if not context:
        return False
    answer_terms = _terms(answer) - _STOP_WORDS - _DOMAIN_TERMS - _QUERY_NOISE
    evidence_terms = set()
    evidence_text = []
    for item in context:
        evidence_terms.update(_terms(item.title))
        evidence_terms.update(_terms(item.content))
        evidence_text.append(_normalize(item.title + " " + item.content))
    joined_evidence = " ".join(evidence_text)
    if any(_phrase(phrase) not in joined_evidence for phrase in _capitalized_phrases(answer)):
        return False
    if len(answer_terms) < 5:
        return bool(answer_terms & evidence_terms)
    supported = answer_terms & evidence_terms
    required = max(3, math.ceil(len(answer_terms) * 0.2))
    if len(supported) < required:
        return False
    sentences = [part.strip() for part in re.split(r"[.!?\n]+", answer) if part.strip()]
    for sentence in sentences:
        terms = _terms(sentence) - _STOP_WORDS - _DOMAIN_TERMS - _QUERY_NOISE
        if len(terms) >= 8 and len(terms & evidence_terms) < 2:
            return False
    for match in re.finditer(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$", answer):
        terms = _terms(match.group(1)) - _STOP_WORDS - _DOMAIN_TERMS - _QUERY_NOISE
        if len(terms) >= 2 and len(terms & evidence_terms) / len(terms) < 0.4:
            return False
    return True


def answer_addresses_question(question, answer):
    """Check a few explicit answer shapes that lexical grounding alone misses."""
    normalized_question = _normalize(question)
    normalized_answer = _normalize(answer)
    asks_type_comparison = (
        any(term in normalized_question for term in ("khac nhau", "khac gi", "khac biet", "phan biet", "so sanh", "the nao"))
        and "type 1" in normalized_question
        and "type 2" in normalized_question
    )
    if asks_type_comparison:
        has_type_1 = any(label in normalized_answer for label in ("type 1", "tuyp 1", "tup 1", "loai 1"))
        has_type_2 = any(label in normalized_answer for label in ("type 2", "tuyp 2", "tup 2", "loai 2"))
        type_1_mechanism = any(term in normalized_answer for term in (
            "tu mien", "mien dich", "khong san xuat insulin", "khong the san xuat insulin",
            "san xuat it insulin", "san xuat rat it insulin", "tuy san xuat rat it insulin",
            "thieu insulin", "te bao beta", "tuy khong san xuat",
        ))
        type_2_mechanism = any(term in normalized_answer for term in (
            "khang insulin", "de khang insulin", "insulin resistance", "su dung insulin khong hieu qua",
            "khong dung insulin hieu qua", "dap ung insulin kem", "khong su dung insulin tot",
        ))
        if any(phrase in normalized_answer for phrase in (
            "de khang insulin trong cac te bao tuyen tuy",
            "te bao de khang insulin trong cac te bao tuyen tuy",
        )):
            return False
        return has_type_1 and has_type_2 and type_1_mechanism and type_2_mechanism

    asks_for_diabetes_types = (
        any(term in normalized_question for term in ("loai nao", "may loai", "nhung loai", "cac loai"))
        and any(term in normalized_question for term in ("tieu duong", "dai thao duong", "diabetes"))
        and not any(term in normalized_question for term in ("thuc pham", "thuc don", "bua an", "nen an", "dinh duong"))
    )
    if asks_for_diabetes_types:
        categories = (
            ("type 1", "tuyp 1", "tup 1", "loai 1"),
            ("type 2", "tuyp 2", "tup 2", "loai 2"),
            ("thai ky", "thai nghen", "thai phu"),
        )
        named = sum(any(label in normalized_answer for label in group) for group in categories)
        return named >= 2

    # Check that the response addresses the requested intent, not merely the
    # same broad topic. These markers are intentionally broad enough to accept
    # short paraphrases while rejecting a related but off-target paragraph.
    intent_markers = (
        (("an gi", "nen an", "che do an", "dinh duong", "thuc pham nao", "loai thuc pham"),
         ("rau", "thuc pham", "bua an", "khau phan", "ngu coc", "chat xo", "dinh duong", "an uong")),
        (("han che an", "nen tranh", "can tranh", "khong nen an"),
         ("han che", "nen tranh", "can tranh", "giam", "thuc pham", "do ngot", "duong")),
        (("trieu chung", "dau hieu", "bieu hien"),
         ("trieu chung", "dau hieu", "khac nuoc", "tieu nhieu", "met moi", "sut can", "mo mat", "te bi")),
        (("bien chung", "tac hai", "hau qua", "nguy hiem"),
         ("bien chung", "ton thuong", "tim mach", "than", "mat", "than kinh", "vo mac", "dot quy")),
        (("nguyen nhan", "vi sao", "tai sao", "do dau"),
         ("nguyen nhan", "khang insulin", "tu mien", "yeu to nguy co", "tuyen tuy", "te bao beta")),
        (("phong ngua", "phong tranh", "ngan ngua"),
         ("phong ngua", "phong tranh", "kiem soat can nang", "van dong", "loi song", "che do an")),
        (("dieu tri", "kiem soat", "chua benh"),
         ("dieu tri", "kiem soat", "thuoc", "insulin", "theo doi", "bac si", "van dong")),
    )
    for question_phrases, required_markers in intent_markers:
        if any(phrase in normalized_question for phrase in question_phrases):
            if not any(marker in normalized_answer for marker in required_markers):
                return False
            asks_what_to_eat = any(phrase in normalized_question for phrase in (
                "an gi", "nen an", "che do an", "thuc pham nao", "loai thuc pham",
            ))
            asks_what_to_limit = any(phrase in normalized_question for phrase in (
                "han che an", "nen tranh", "can tranh", "khong nen an",
            ))
            if asks_what_to_eat and not asks_what_to_limit and any(term in normalized_answer for term in (
                "han che", "nen tranh", "khong nen", "can tranh",
            )):
                return False
            if asks_what_to_eat and not asks_what_to_limit and any(term in normalized_answer for term in (
                "che bien", "chien", "ran", "xao",
            )) and not any(term in normalized_question for term in ("che bien", "nau", "mon an")):
                return False
            if asks_what_to_eat and not asks_what_to_limit:
                food_answer = normalized_answer.split(".", 1)[0]
                asks_meal_plan = any(phrase in normalized_question for phrase in (
                    "thuc don", "bua sang", "bua trua", "bua toi",
                ))
                acknowledges_missing_menu = any(phrase in normalized_answer for phrase in (
                    "chua du chi tiet de lap thuc don", "chua du co so de lap thuc don",
                ))
                if asks_meal_plan and not acknowledges_missing_menu and not all(
                    meal in normalized_answer for meal in ("bua sang", "bua trua", "bua toi")
                ):
                    return False
                food_groups = (
                    ("rau",), ("trai cay",), ("ngu coc",), ("sua",),
                    ("thit nac",), ("gia cam",), ("ca",), ("dau",),
                )
                mentioned_groups = sum(
                    any(re.search(rf"\b{re.escape(term)}\b", food_answer) for term in group)
                    for group in food_groups
                )
                if mentioned_groups > 3:
                    return False
            return True

    question_terms = _terms(question) - _STOP_WORDS - _DOMAIN_TERMS - _QUERY_NOISE - {
        "khac", "biet", "sanh", "phan", "anh", "huong", "theo", "nao", "duoc", "nen",
        "bao", "nhieu", "la", "gi", "co", "khong", "mot", "nhung", "cac", "ve",
        "tai", "sao", "vi", "do", "dau", "the", "the", "nao", "giua", "hay",
    }
    question_terms = {term for term in question_terms if len(term) >= 3}
    if not question_terms:
        return True
    answer_terms = _terms(answer)
    overlap = question_terms & answer_terms
    required = min(2, len(question_terms))
    return len(overlap) >= required


def _terms(text):
    normalized = unicodedata.normalize("NFD", str(text or "").casefold())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.replace("đ", "d")
    return set(re.findall(r"[a-z0-9]+", normalized))


def _normalize(text):
    normalized = unicodedata.normalize("NFD", str(text or "").casefold())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", normalized.replace("đ", "d")).strip()


def _phrase(text):
    return _normalize(text)


def _capitalized_phrases(text):
    phrases = []
    for match in re.finditer(r"\b([A-ZÀ-Ỵ][\wÀ-ỹ]*\s+[A-ZÀ-Ỵ][\wÀ-ỹ]*)\b", str(text or "")):
        phrase = match.group(1)
        if len(phrase) > 2:
            phrases.append(phrase)
    return phrases


_STOP_WORDS = {
    "la", "gi", "sao", "nao", "the", "nao", "co", "cua", "va", "hay",
    "cho", "toi", "voi", "mot", "nhung", "cac", "nen", "can", "duoc",
    "nhu", "the", "ve", "trong", "khi", "tai", "do", "giua", "loai",
    "benh", "tieu", "duong", "dai", "thao", "duong", "diabetes", "type",
    "tuyp", "tup", "la", "khong", "nguoi", "mac", "bi",
}
_DOMAIN_TERMS = {"diabetes", "insulin", "glucose", "duonghuyet", "gi", "gl", "glycemic", "index", "load"}
_QUERY_NOISE = {"y", "luu", "chu", "nghi", "thay", "roi", "xem"}
