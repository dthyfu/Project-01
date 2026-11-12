import re
import logging
from time import perf_counter

from agents.base import is_non_vietnamese
from agents.data_analysis import (
    DataAnalysisAgent,
    analyze_inline_fasting_glucose_series,
    answer_inline_fasting_glucose,
    summarize_numeric_table,
)
from agents.knowledge import KnowledgeAgent
from agents.medical_research import MedicalResearchAgent
from agents.nutrition import NutritionAgent
from generation import generate_answer
from schemas.messages import AgentResult, ChatResponse
from services.retrieval_service import (
    answer_addresses_question,
    answer_supported_by_context,
    context_supports_question,
    filter_relevant_chunks,
)
from services.ollama_client import OllamaError

logger = logging.getLogger(__name__)


OUT_OF_SCOPE = "Mình hiện hỗ trợ câu hỏi nghiên cứu về bệnh tiểu đường và phân tích dữ liệu bạn cung cấp."
INSUFFICIENT = "Tài liệu hiện có chưa đủ thông tin để kết luận chính xác về câu hỏi này."
OLLAMA_FALLBACK = "Ollama hiện không khả dụng và tài liệu chưa đủ cơ sở để tạo câu trả lời an toàn."


class Supervisor:
    """Rule-routed orchestrator. Specialists share one retrieval context."""

    def __init__(self, llm_client, retrieval_service, answer_fallback=generate_answer):
        self.llm = llm_client
        self.retrieval = retrieval_service
        self.answer_fallback = answer_fallback
        self.agents = {
            "medical": MedicalResearchAgent(llm_client),
            "nutrition": NutritionAgent(llm_client),
            "data_analysis": DataAnalysisAgent(llm_client),
            "knowledge": KnowledgeAgent(llm_client),
        }

    def process(self, question, history=None, data_text="", emit=None):
        process_started = perf_counter()
        routing_started = process_started
        emit = emit or (lambda event, payload: None)
        emit("status", {"agent": "supervisor", "status": "running", "message": "Đang phân tích câu hỏi"})
        history = history or []
        effective_question = _resolve_follow_up(question, history)
        safety_answer = _personal_medical_guardrail(effective_question, history)
        if safety_answer:
            emit("status", {"agent": "supervisor", "status": "completed", "message": "Đã áp dụng giới hạn an toàn y tế"})
            return ChatResponse(safety_answer, [], "completed", {"provider": "safety_rule", "llm_calls": 0})
        glucose_series_answer = analyze_inline_fasting_glucose_series(effective_question)
        if glucose_series_answer:
            emit("status", {"agent": "data_analysis", "status": "completed", "message": "Đã tính và phân loại các giá trị được cung cấp"})
            return ChatResponse(
                glucose_series_answer, ["data_analysis"], "completed",
                {"provider": "local_analysis", "llm_calls": 0,
                 "reference_source": "NIDDK",
                 "reference_url": "https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis"},
            )
        threshold_answer = answer_inline_fasting_glucose(effective_question)
        if threshold_answer:
            emit("status", {"agent": "medical", "status": "completed", "message": "Đã đối chiếu kết quả với ngưỡng xét nghiệm tham chiếu"})
            return ChatResponse(
                threshold_answer, ["medical"], "completed",
                {"provider": "clinical_threshold_rule", "llm_calls": 0,
                 "reference_source": "NIDDK",
                 "reference_url": "https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis"},
            )
        plan = self._plan(effective_question, data_text, history)
        routing_ms = round((perf_counter() - routing_started) * 1000)
        if plan["out_of_scope"]:
            emit("status", {"agent": "supervisor", "status": "completed", "message": "Đã xác định phạm vi câu hỏi"})
            return ChatResponse(OUT_OF_SCOPE, [], "completed")
        context = []
        stage_timings = {"routing": routing_ms}
        retrieval_error = None
        if plan["needs_retrieval"]:
            emit("status", {"agent": "retrieval", "status": "running", "message": "Đang tìm kiếm kiến thức"})
            stage_started = perf_counter()
            try:
                raw_context = self.retrieval.search(effective_question, top_k=8)
                context = filter_relevant_chunks(effective_question, raw_context)
                if not context_supports_question(effective_question, context):
                    context = []
                logger.info(
                    "retrieval context raw_count=%d relevant_count=%d selected_ids=%s",
                    len(raw_context), len(context), [item.source_id for item in context],
                )
                emit("status", {
                    "agent": "retrieval",
                    "status": "completed" if context else "insufficient",
                    "message": "Đã tìm thấy tài liệu phù hợp" if context else "Chưa tìm thấy tài liệu đủ liên quan",
                })
            except Exception:
                retrieval_error = "retrieval_failed"
                emit("status", {"agent": "retrieval", "status": "failed", "message": "Không thể truy xuất tài liệu"})
            finally:
                stage_timings["retrieval"] = round((perf_counter() - stage_started) * 1000)

        results = []
        agents_used = []
        errors = []
        ollama_error = False
        if plan["agents"]:
            emit("status", {"agent": "supervisor", "status": "running", "message": "Đang tạo câu trả lời"})
        for name in plan["agents"]:
            if name != "data_analysis" and not context and not (name == "knowledge" and data_text and not plan["needs_retrieval"]):
                continue
            if name == "data_analysis" and not summarize_numeric_table(data_text):
                emit("status", {"agent": name, "status": "skipped", "message": "Chưa có bảng dữ liệu để phân tích"})
                continue
            emit("status", {"agent": name, "status": "running", "message": _agent_label(name) + " đang xử lý"})
            stage_started = perf_counter()
            try:
                extra = {"user_supplied_content": data_text[:6000]} if data_text else None
                if name == "data_analysis":
                    agent_result = self.agents[name].run(
                        effective_question, context, history, data_text=data_text
                    )
                else:
                    agent_result = self.agents[name].run(
                        effective_question, context, history, extra=extra
                    )
                if is_non_vietnamese(agent_result):
                    raise ValueError("Model trả lời bằng ngôn ngữ không phù hợp.")
                if _requires_gi_gl(effective_question) and not _mentions_gi_gl(agent_result):
                    raise ValueError("Câu trả lời bỏ sót một phần GI/GL được hỏi.")
                if name != "data_analysis" and context and not answer_supported_by_context(agent_result, context):
                    raise ValueError("Câu trả lời không khớp đủ với tài liệu truy xuất.")
                if name != "data_analysis" and not answer_addresses_question(effective_question, agent_result):
                    raise ValueError("Câu trả lời chưa nêu đủ các loại bệnh được hỏi.")
                results.append(AgentResult(
                    agent=name,
                    status="completed",
                    answer=agent_result,
                    source_ids=[item.source_id for item in context],
                ))
                agents_used.append(name)
                emit("status", {"agent": name, "status": "completed", "message": _agent_label(name) + " đã hoàn tất"})
            except Exception as error:
                errors.append(name)
                ollama_error = ollama_error or isinstance(error, OllamaError)
                logger.exception(
                    "agent failed agent=%s error_type=%s ollama_error=%s",
                    name, type(error).__name__, isinstance(error, OllamaError),
                )
                results.append(AgentResult(agent=name, status="failed", error=type(error).__name__))
                emit("status", {"agent": name, "status": "failed", "message": _agent_label(name) + " gặp lỗi; đang dùng phương án dự phòng"})
            finally:
                stage_timings[f"agent_{name}"] = round((perf_counter() - stage_started) * 1000)

        successful = [item for item in results if item.status == "completed" and item.answer]
        if successful:
            answer = _combine_answers([item.answer for item in successful])
            if _contains_prescriptive_dose(answer):
                answer = "Mình không thể khuyến nghị liều thuốc hoặc thay đổi điều trị. Hãy trao đổi với bác sĩ hoặc dược sĩ đang theo dõi bạn."
            status = "partial" if errors or retrieval_error or (plan["needs_retrieval"] and not context) else "completed"
            provider = "ollama"
        elif plan["agents"] == ["data_analysis"]:
            summary = summarize_numeric_table(data_text)
            answer = _format_data_summary(summary) if summary else "Bạn hãy gửi bảng dữ liệu có tiêu đề cột và ít nhất hai dòng số liệu để mình phân tích."
            status = "fallback" if summary else "insufficient"
            provider = "local_analysis"
        elif context:
            answer = self.answer_fallback(effective_question, [
                {"content": item.content, "title": item.title, "url": item.url}
                for item in context
            ])
            if not answer_addresses_question(effective_question, answer):
                answer = INSUFFICIENT
                status = "insufficient"
                provider = "none"
            elif _requires_gi_gl(effective_question) and not _mentions_gi_gl(answer):
                answer = INSUFFICIENT
                status = "insufficient"
                provider = "none"
            elif _is_insufficient(answer):
                answer = OLLAMA_FALLBACK if ollama_error else INSUFFICIENT
                status = "error" if errors else "insufficient"
            else:
                status = "fallback"
            provider = "rule_fallback"
        else:
            answer = OLLAMA_FALLBACK if errors and not retrieval_error else INSUFFICIENT
            status = "error" if errors and not retrieval_error else "insufficient"
            provider = "none"

        if errors:
            logger.warning(
                "supervisor fallback provider=%s status=%s failed_agents=%s ollama_error=%s retrieval_error=%s",
                provider, status, errors, ollama_error, retrieval_error,
            )

        emit("status", {"agent": "supervisor", "status": "completed", "message": "Đã hoàn tất câu trả lời"})
        stage_timings["supervisor_total"] = round((perf_counter() - process_started) * 1000)
        return ChatResponse(
            answer=_clean_final(answer),
            agents_used=agents_used,
            status=status,
            metadata={
                "provider": provider,
                "llm_calls": len(results),
                "retrieved_count": len(context),
                "source_ids": [item.source_id for item in context],
                "source_metadata": [
                    {"source_id": item.source_id, "title": item.title, "url": item.url}
                    for item in context
                ],
                "agent_results": [item.to_dict() for item in results],
                "errors": errors,
                "retrieval_error": retrieval_error,
                "timings_ms": stage_timings,
            },
        )

    def _plan(self, question, data_text, history=None):
        history = history or []
        lowered = question.casefold()
        data_requested = any(term in lowered for term in (
            "phân tích dữ liệu", "phân tích bảng", "xu hướng dữ liệu", "số liệu này",
            "tính giá trị trung bình", "tính trung bình", "xác định số người",
            "phân bố dữ liệu", "đối chiếu với ngưỡng", "ngưỡng tham chiếu",
        ))
        data_present = bool(data_text and summarize_numeric_table(data_text))
        agents = []
        nutrition = any(term in lowered for term in (
            "dinh dưỡng", "chế độ ăn", "ăn gì", "thực phẩm", "bữa ăn", "bữa sáng",
            "đường huyết sau ăn", "uống gì", "ăn được", "nên ăn", "hạn chế ăn",
            "chuối", "chỉ số gi", "chỉ số gl", "glycemic index", "glycemic load",
        ))
        diabetes = any(term in lowered for term in (
            "tiểu đường", "đái tháo đường", "diabetes", "insulin", "đường huyết", "glucose",
            "chỉ số gi", "chỉ số gl", "glycemic index", "glycemic load",
        ))
        medical_topic = any(term in lowered for term in (
            "triệu chứng", "dấu hiệu", "nguyên nhân", "cơ chế", "biến chứng",
            "điều trị", "chữa bệnh", "phòng ngừa", "ảnh hưởng", "chẩn đoán",
            "khác nhau", "định nghĩa",
        ))
        in_diabetes_history = any(
            any(term in str(message.get("content", "")).casefold() for term in (
                "tiểu đường", "đái tháo đường", "diabetes", "insulin", "đường huyết",
            )) for message in history[-6:]
        )
        data_only = data_requested or data_present
        if data_only:
            agents.append("data_analysis")
        if nutrition:
            agents.append("nutrition")
        if (diabetes or in_diabetes_history) and (medical_topic or not nutrition):
            agents.append("medical")
        if "medical" in agents and "nutrition" in agents:
            agents.remove("medical")
            agents.insert(agents.index("nutrition"), "medical")
        relation = any(term in lowered for term in (
            "mối quan hệ", "liên quan như thế nào", "kết hợp",
        ))
        if relation and (diabetes or nutrition or in_diabetes_history):
            agents.append("knowledge")
        agents = list(dict.fromkeys(agents))
        explicit_outside = any(term in lowered for term in (
            "cổ phiếu", "chứng khoán", "giá vàng", "thời tiết", "bầu cử", "bitcoin",
        )) and not diabetes
        if not agents and not data_only and not in_diabetes_history:
            explicit_outside = True
        if data_only and not data_present and not data_requested:
            agents = ["data_analysis"]
        return {
            "agents": agents,
            "needs_retrieval": any(name in {"medical", "nutrition", "knowledge"} for name in agents) and (
                diabetes or in_diabetes_history or any(name in {"medical", "nutrition"} for name in agents)
            ),
            "out_of_scope": explicit_outside,
        }


def _resolve_follow_up(question, history):
    if history and any(term in question.casefold() for term in ("còn", "vậy", "thế", "chúng", "đó", "nó")):
        prior = " ".join(str(item.get("content", "")) for item in history[-4:])
        if any(term in prior.casefold() for term in ("tiểu đường", "đái tháo đường", "diabetes")):
            return f"Bệnh tiểu đường: {question}"
    return question


def _combine_answers(answers):
    unique = []
    seen = set()
    for answer in answers:
        answer = answer.strip()
        key = re.sub(r"\W+", " ", answer.casefold()).strip()
        if answer and key not in seen:
            unique.append(answer)
            seen.add(key)
    return "\n\n".join(unique)


def _clean_final(answer):
    answer = re.sub(r"https?://\S+", "", answer or "", flags=re.IGNORECASE)
    answer = re.sub(r"(?im)^.*(?:nguồn tham khảo|tài liệu tham khảo|sources?:).*?$", "", answer)
    answer = re.sub(r"\[(?:source|chunk)\s*:?\s*[^\]]+\]", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\[(?:\d+)\]", "", answer)
    answer = re.sub(r"\b(?:source|chunk)_?id\s*[:=]\s*[\w-]+", "", answer, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", answer).strip()


def _is_insufficient(answer):
    text = (answer or "").casefold()
    return any(term in text for term in (
        "chưa tìm thấy", "chưa cung cấp đủ", "chưa đủ thông tin", "chưa tìm thấy nội dung",
    ))


def _agent_label(name):
    return {
        "medical": "Medical Agent",
        "nutrition": "Nutrition Agent",
        "data_analysis": "Data Analysis Agent",
        "knowledge": "Knowledge Agent",
    }.get(name, name)


def _format_data_summary(summary):
    if not summary:
        return "Bạn hãy gửi bảng dữ liệu có tiêu đề cột và ít nhất hai dòng số liệu để mình phân tích."
    rows = [
        f"{name}: {values['count']} giá trị; trung bình {values['mean']}, "
        f"thấp nhất {values['min']}, cao nhất {values['max']}"
        for name, values in summary.items()
    ]
    return "Tóm tắt thống kê từ dữ liệu bạn gửi: " + "; ".join(rows) + "."


def _personal_medical_guardrail(question, history=None):
    text = question.casefold()
    history = history or []
    personal_diagnosis = any(phrase in text for phrase in (
        "tôi có bị", "tôi bị tiểu đường", "tôi mắc tiểu đường", "kết quả của tôi",
        "đường huyết của tôi", "có phải tôi mắc", "tôi có mắc", "tôi mắc không",
    ))
    follow_up_about_self = any(
        any(term in str(message.get("content", "")).casefold() for term in (
            "tiểu đường", "đái tháo đường", "diabetes", "đường huyết",
        )) for message in history[-6:]
    ) and "tôi" in text and any(term in text for term in ("có bị", "bị không", "mắc không"))
    medication_change = any(phrase in text for phrase in (
        "tăng liều", "giảm liều", "đổi liều", "bỏ thuốc", "ngưng thuốc",
        "ngừng thuốc", "tự điều chỉnh liều",
    )) or (
        any(term in text for term in ("thuốc", "insulin", "metformin", "sulfonylurea"))
        and any(term in text for term in ("bao nhiêu", "liều", "uống thế nào"))
    )
    if personal_diagnosis or follow_up_about_self:
        return "Mình không thể chẩn đoán cá nhân từ cuộc trò chuyện này. Hãy trao đổi với nhân viên y tế để được đánh giá kết quả và tình trạng cụ thể."
    if medication_change:
        return "Mình không thể hướng dẫn thay đổi hoặc xác định liều thuốc. Đừng tự ý đổi liều; hãy hỏi bác sĩ hoặc dược sĩ đang theo dõi bạn."
    return None


def _contains_prescriptive_dose(answer):
    return bool(re.search(
        r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ml|mL|đơn vị|units?)\b",
        answer or "",
        re.IGNORECASE,
    ))


def _requires_gi_gl(question):
    return bool(re.search(r"\bgi\b", question, re.IGNORECASE) and re.search(r"\bgl\b", question, re.IGNORECASE))


def _mentions_gi_gl(answer):
    text = answer or ""
    return bool(re.search(r"\bgi\b", text, re.IGNORECASE) and
                (re.search(r"\bgl\b", text, re.IGNORECASE) or re.search(r"tải đường huyết", text, re.IGNORECASE)))
