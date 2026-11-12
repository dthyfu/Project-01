import json
import io
import tempfile
import unittest
import zipfile
import base64
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from agents.data_analysis import (
    analyze_inline_fasting_glucose_series,
    answer_inline_fasting_glucose,
    summarize_numeric_table,
)
from agents.supervisor import INSUFFICIENT, Supervisor, _personal_medical_guardrail
from agents.base import _clean_answer, _context_payload
from agents.base import is_non_vietnamese, is_repetitive_or_corrupt
from generation import generate_answer
from schemas.messages import ChatResponse, RetrievedChunk
from services.conversation_store import ConversationStore
from services.diabetes_risk import RiskInputError, predict_risk, validate_input
from services.ollama_client import OllamaClient, OllamaModelNotFound, OllamaUnavailable
from services.ollama_client import OllamaConnectionError, OllamaGenerationError, OllamaRequestTimeout
from services.ollama_client import OllamaResponseError
from services.retrieval_service import RetrievalService, answer_addresses_question, filter_relevant_chunks
from services.retrieval_service import answer_supported_by_context, context_supports_question
from services.attachment_service import AttachmentService, AttachmentError, _extract


class FakeRag:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls = []

    def search(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.chunks


class FakeLlm:
    def __init__(self, answer="Bệnh tiểu đường là một rối loạn chuyển hóa.", error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    def chat(self, system_prompt, user_prompt, num_predict=450):
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return self.answer


def diabetes_chunk():
    return {
        "chunk_id": 7,
        "title": "Bệnh tiểu đường",
        "url": "https://internal.example/article",
        "content": "Bệnh tiểu đường là một rối loạn chuyển hóa. Type 1 và type 2 có đặc điểm khác nhau và có thể gây biến chứng. Dinh dưỡng cân bằng và ăn uống hợp lý có vai trò hỗ trợ.",
        "final_score": 0.05,
    }


class OllamaClientTests(unittest.TestCase):
    def test_chat_uses_local_chat_api_and_options(self):
        client = OllamaClient()
        response = Mock(status_code=200)
        response.json.return_value = {"message": {"content": "Câu trả lời"}}
        with patch("services.ollama_client.requests.post", return_value=response) as post:
            self.assertEqual(client.chat("system", "user"), "Câu trả lời")
        args, kwargs = post.call_args
        self.assertTrue(args[0].endswith("/api/chat"))
        self.assertEqual(kwargs["json"]["model"], "qwen2.5:1.5b")
        self.assertEqual(kwargs["json"]["options"]["num_ctx"], 4096)
        self.assertFalse(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["options"]["repeat_penalty"], 1.15)
        self.assertEqual(kwargs["json"]["options"]["repeat_last_n"], 128)
        self.assertEqual(kwargs["json"]["keep_alive"], "10m")

    def test_non_completed_generation_is_preserved_for_quality_gate(self):
        client = OllamaClient()
        response = Mock(status_code=200)
        response.json.return_value = {
            "message": {"content": "Câu trả lời chưa hoàn tất"},
            "done": False, "done_reason": "length", "eval_count": 450,
        }
        with patch("services.ollama_client.requests.post", return_value=response):
            completion = client.chat_with_metadata("system", "user")
        self.assertFalse(completion.done)
        self.assertEqual(completion.done_reason, "length")
        self.assertEqual(completion.output_tokens, 450)

    def test_missing_model_and_connection_error_are_clear(self):
        client = OllamaClient()
        missing = Mock(status_code=404)
        with patch("services.ollama_client.requests.post", return_value=missing):
            with self.assertRaises(OllamaModelNotFound):
                client.chat("system", "user")
        with patch("services.ollama_client.requests.post", side_effect=requests.ConnectionError()):
            with self.assertRaises(OllamaUnavailable):
                client.chat("system", "user")

    def test_timeout_and_malformed_response_are_clear(self):
        client = OllamaClient()
        with patch("services.ollama_client.requests.post", side_effect=requests.Timeout()):
            with self.assertRaises(OllamaRequestTimeout):
                client.chat("system", "user")
        response = Mock(status_code=200)
        response.json.return_value = {"unexpected": "shape"}
        with patch("services.ollama_client.requests.post", return_value=response):
            with self.assertRaises(OllamaResponseError):
                client.chat("system", "user")

    def test_health_separates_server_installed_and_loaded_model(self):
        client = OllamaClient()
        tags = Mock(status_code=200)
        tags.json.return_value = {"models": [{"name": "qwen2.5:1.5b"}]}
        ps = Mock(status_code=200)
        ps.json.return_value = {"models": []}
        with patch("services.ollama_client.requests.get", side_effect=[tags, ps]):
            state = client.health()
        self.assertEqual(state["server"], "available")
        self.assertTrue(state["model_installed"])
        self.assertFalse(state["model_loaded"])
        self.assertEqual(state["model_state"], "unloaded")

    def test_connection_and_http_generation_errors_are_distinct(self):
        client = OllamaClient()
        with patch("services.ollama_client.requests.post", side_effect=requests.ConnectionError()):
            with self.assertRaises(OllamaConnectionError):
                client.chat("system", "user")
        response = Mock(status_code=500, text="runner failed")
        with patch("services.ollama_client.requests.post", return_value=response):
            with self.assertRaises(OllamaGenerationError):
                client.chat("system", "user")


class RetrievalAndSupervisorTests(unittest.TestCase):
    def test_answer_relevance_gate_rejects_same_topic_but_wrong_intent(self):
        self.assertFalse(answer_addresses_question(
            "Bệnh tiểu đường gây biến chứng gì?",
            "Người bệnh nên ưu tiên rau và chất xơ trong chế độ ăn.",
        ))
        self.assertTrue(answer_addresses_question(
            "Bệnh tiểu đường gây biến chứng gì?",
            "Biến chứng có thể ảnh hưởng đến mắt, thận và tim mạch.",
        ))
        self.assertFalse(answer_addresses_question(
            "Người mắc bệnh tiểu đường nên ăn gì?",
            "Type 2 thường liên quan đến tình trạng kháng insulin.",
        ))
        self.assertFalse(answer_addresses_question(
            "Người mắc bệnh tiểu đường nên ăn gì?",
            "Có thể ưu tiên rau và chất xơ. Nên hạn chế đồ ngọt.",
        ))
        self.assertFalse(answer_addresses_question(
            "Người mắc bệnh tiểu đường nên ăn gì?",
            "Nên ưu tiên rau, trái cây, ngũ cốc và cá; chế biến bằng cách chiên hoặc xào.",
        ))

    def test_type_comparison_prompt_keeps_only_mechanism_evidence(self):
        chunks = [
            RetrievedChunk("chunk:noise", "noise", "Bệnh tiểu đường", "", "Bài viết về quế và HbA1c.", 0.99),
            RetrievedChunk("chunk:type1", "type1", "Type 1", "", "Type 1 do tự miễn, tế bào beta giảm sản xuất insulin.", 0.7),
            RetrievedChunk("chunk:type2", "type2", "Type 2", "", "Type 2 liên quan đến đề kháng insulin.", 0.6),
        ]
        payload = _context_payload(
            chunks, "Tiểu đường type 1 và type 2 khác nhau thế nào?"
        )
        self.assertEqual([item["source_id"] for item in payload], ["chunk:type1", "chunk:type2"])
        self.assertLessEqual(sum(len(item["content"]) for item in payload), 2400)

    def test_retrieval_diversifies_context_across_article_urls(self):
        chunks = [
            {**diabetes_chunk(), "chunk_id": 1, "url": "https://a.example/", "content": "a1"},
            {**diabetes_chunk(), "chunk_id": 2, "url": "https://a.example/", "content": "a2"},
            {**diabetes_chunk(), "chunk_id": 3, "url": "https://a.example/", "content": "a3"},
            {**diabetes_chunk(), "chunk_id": 4, "url": "https://b.example/", "content": "b1"},
        ]
        retrieved = RetrievalService(FakeRag(chunks)).search("tiểu đường", top_k=3)
        self.assertEqual([item.chunk_id for item in retrieved], ["1", "4", "2"])

    def test_type_comparison_requires_both_types_and_distinguishing_mechanisms(self):
        question = "Tiểu đường type 1 và type 2 khác nhau thế nào?"
        cinnamon_only = (
            "Type 2 có thể được cải thiện bằng quế, làm giảm HbA1c và tăng độ nhạy insulin."
        )
        complete = (
            "Type 1 thường do tự miễn làm giảm sản xuất insulin; type 2 thường liên quan đến kháng insulin."
        )
        self.assertFalse(answer_addresses_question(question, cinnamon_only))
        self.assertTrue(answer_addresses_question(question, complete))
        self.assertFalse(answer_addresses_question(
            question,
            "Type 1: miễn dịch tấn công tuyến tụy. Type 2: tế bào đề kháng insulin trong các tế bào tuyến tụy không đáp ứng insulin.",
        ))

    def test_type_comparison_filters_chunks_about_type2_cinnamon(self):
        question = "Tiểu đường type 1 và type 2 khác nhau thế nào?"
        raw = RetrievalService(FakeRag([
            {**diabetes_chunk(), "chunk_id": 1,
             "url": "https://a.example/type1",
             "content": "Type 1 thường do tự miễn làm giảm sản xuất insulin."},
            {**diabetes_chunk(), "chunk_id": 2,
             "url": "https://b.example/cinnamon",
             "content": "Một nghiên cứu dùng quế ở người mắc type 2 theo dõi HbA1c."},
            {**diabetes_chunk(), "chunk_id": 3,
             "url": "https://c.example/type2",
             "content": "Type 2 thường liên quan đến đề kháng insulin."},
        ])).search(question, top_k=3)
        relevant = filter_relevant_chunks(question, raw)
        self.assertEqual([item.chunk_id for item in relevant], ["1", "3"])

    def test_comparison_fallback_collects_both_types_from_separate_chunks(self):
        question = "Tiểu đường type 1 và type 2 khác nhau thế nào?"
        chunks = [
            {**diabetes_chunk(), "chunk_id": 1,
             "content": "Tiểu đường type 1 thường do tự miễn làm giảm sản xuất insulin."},
            {**diabetes_chunk(), "chunk_id": 2,
             "content": "Tiểu đường type 2 thường liên quan đến kháng insulin."},
        ]
        answer = generate_answer(question, chunks)
        self.assertTrue(answer_addresses_question(question, answer))

    def test_type_comparison_fallback_summarizes_both_supported_mechanisms(self):
        answer = generate_answer("Tiểu đường type 1 và type 2 khác nhau thế nào?", [
            {"content": "Type 1 do tự miễn làm giảm sản xuất insulin."},
            {"content": "Type 2 liên quan đến đề kháng insulin."},
        ])
        self.assertIn("tự miễn", answer)
        self.assertIn("đề kháng insulin", answer)

    def test_nutrition_fallback_selects_recommendations_not_limit_or_drink_intro(self):
        answer = generate_answer("Người mắc bệnh tiểu đường nên ăn gì?", [
            {"title": "Đồ uống", "content": "Người bệnh nên hạn chế đồ uống có nhiều đường."},
            {"title": "Dinh dưỡng", "content": (
                "Bệnh nhân tiểu đường nên ăn thực phẩm giàu chất xơ. "
                "Các lựa chọn phù hợp gồm rau, trái cây, ngũ cốc nguyên hạt, cá và đậu."
            )},
        ])
        self.assertIn("rau", answer.casefold())
        self.assertNotIn("hạn chế", answer.casefold())

    def test_food_recommendation_fallback_summarizes_evidence_concisely(self):
        documents = [{"title": "Thực phẩm", "content": (
            "Bệnh nhân tiểu đường nên ăn thực phẩm giàu chất xơ. "
            "Lựa chọn thực phẩm tốt gồm rau xanh, trái cây, ngũ cốc nguyên hạt, cá và đậu."
        )}]
        answer = generate_answer("Người mắc bệnh tiểu đường nên ăn gì?", documents)
        self.assertEqual(answer, "Có thể ưu tiên rau xanh, trái cây và ngũ cốc nguyên hạt.")
        self.assertTrue(answer_addresses_question("Người mắc bệnh tiểu đường nên ăn gì?", answer))

    def test_food_fallback_discloses_missing_support_for_a_three_meal_menu(self):
        answer = generate_answer(
            "Người mắc bệnh tiểu đường type 2 nên lựa chọn thực phẩm nào? "
            "Hãy đề xuất thực đơn bữa sáng, bữa trưa và bữa tối.",
            [{"content": "Người bệnh nên ăn thực phẩm giàu chất xơ như rau xanh, trái cây và ngũ cốc nguyên hạt."}],
        )
        self.assertIn("rau xanh", answer)
        self.assertIn("chưa đủ chi tiết để lập thực đơn", answer)
        self.assertTrue(answer_addresses_question(
            "Người bệnh nên lựa chọn thực phẩm nào và đề xuất thực đơn bữa sáng, bữa trưa, bữa tối?", answer
        ))

    def test_supervisor_gives_supported_food_groups_and_discloses_missing_menu_evidence(self):
        chunk = diabetes_chunk()
        chunk["content"] = (
            "Người bệnh tiểu đường type 2 nên ăn thực phẩm giàu chất xơ. "
            "Lựa chọn thực phẩm phù hợp gồm rau xanh, trái cây và ngũ cốc nguyên hạt."
        )
        llm = FakeLlm("HbA1c là chỉ số quan trọng đối với người bệnh tiểu đường.")
        question = (
            "Người mắc bệnh tiểu đường type 2 nên lựa chọn những loại thực phẩm nào? "
            "Hãy đề xuất thực đơn mẫu cho bữa sáng, bữa trưa và bữa tối."
        )
        result = Supervisor(llm, RetrievalService(FakeRag([chunk]))).process(question)
        self.assertEqual(result.status, "fallback")
        self.assertIn("rau xanh, trái cây và ngũ cốc nguyên hạt", result.answer)
        self.assertIn("chưa đủ chi tiết để lập thực đơn", result.answer)

    def test_food_recommendation_supervisor_uses_evidence_fallback_if_ollama_fails(self):
        chunk = diabetes_chunk()
        chunk["content"] = (
            "Bệnh nhân tiểu đường nên ăn thực phẩm giàu chất xơ. "
            "Lựa chọn thực phẩm tốt gồm rau xanh, trái cây, ngũ cốc nguyên hạt, cá và đậu."
        )
        result = Supervisor(
            FakeLlm(error=OllamaUnavailable("offline")),
            RetrievalService(FakeRag([chunk])),
        ).process("Người mắc bệnh tiểu đường nên ăn gì?")
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.answer, "Có thể ưu tiên rau xanh, trái cây và ngũ cốc nguyên hạt.")
        self.assertEqual(result.agents_used, [])

    def test_types_question_rejects_retrieved_intro_that_does_not_name_types(self):
        question = "Bệnh tiểu đường có những loại nào?"
        irrelevant = (
            "Tuy vậy, không phải ai cũng biết về các loại bệnh đái tháo đường "
            "và loại nào có nguy cơ cao nhất, cùng tìm hiểu thông tin qua bài viết này."
        )
        self.assertFalse(answer_addresses_question(question, irrelevant))
        self.assertTrue(answer_addresses_question(
            question, "Tiểu đường type 1 và type 2 là hai loại thường được phân biệt."
        ))

    def test_types_question_abstains_if_both_agent_and_fallback_are_off_topic(self):
        question = "Bệnh tiểu đường có những loại nào?"
        chunk = diabetes_chunk()
        chunk["content"] = (
            "Tuy vậy, không phải ai cũng biết về các loại bệnh đái tháo đường "
            "và loại nào có nguy cơ cao nhất, cùng tìm hiểu về nội dung bài viết này."
        )
        llm = FakeLlm(
            "Tuy vậy, không phải ai cũng biết về các loại bệnh đái tháo đường "
            "và loại nào có nguy cơ cao nhất, cùng tìm hiểu thông tin qua bài viết này."
        )
        fallback = Mock(return_value=chunk["content"])
        result = Supervisor(llm, RetrievalService(FakeRag([chunk])), fallback).process(question)
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.answer, INSUFFICIENT)

    def test_types_fallback_returns_only_the_supported_categories(self):
        answer = generate_answer("Bệnh tiểu đường có những loại nào?", [{
            "content": "Ba dạng thường gặp là type 1, type 2 và tiểu đường thai kỳ. "
                       "Nghiên cứu về tiêu thụ đường và type 2 đang được tiến hành."
        }])
        self.assertEqual(
            answer,
            "Các dạng thường gặp được tài liệu nêu gồm type 1, type 2, đái tháo đường thai kỳ.",
        )
        self.assertFalse(answer.endswith("đang được tiến hành."))

    def test_supervisor_replaces_article_intro_with_direct_types_answer(self):
        chunk = diabetes_chunk()
        chunk["content"] = (
            "Ba dạng thường gặp là type 1, type 2 và tiểu đường thai kỳ. "
            "Nghiên cứu về mối liên hệ giữa tiêu thụ đường và type 2 đang được tiến hành."
        )
        llm = FakeLlm(
            "Nghiên cứu về mối liên hệ giữa tiêu thụ đường và bệnh tiểu đường type 2 đang được tiến hành."
        )
        result = Supervisor(llm, RetrievalService(FakeRag([chunk]))).process(
            "Bệnh tiểu đường có những loại nào?"
        )
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.agents_used, [])
        self.assertEqual(
            result.answer,
            "Các dạng thường gặp được tài liệu nêu gồm type 1, type 2, đái tháo đường thai kỳ.",
        )
        self.assertNotIn("nghiên cứu", result.answer.casefold())

    def test_inline_fasting_glucose_and_a1c_results_are_interpreted_safely(self):
        answer = answer_inline_fasting_glucose(
            "Tôi 25 tuổi, glucose lúc đói là 132 mg/dL và HbA1c là 6,8%. Có bình thường không?"
        )
        self.assertIn("132 mg/dL thuộc trong ngưỡng đái tháo đường", answer)
        self.assertIn("6,8% thuộc trong ngưỡng đái tháo đường", answer)
        self.assertIn("không tự xác lập chẩn đoán", answer)

    def test_inline_glucose_series_computes_mean_and_threshold_counts(self):
        answer = analyze_inline_fasting_glucose_series(
            "Dữ liệu đường huyết lúc đói lần lượt là 95, 110, 128, 145 và 160 mg/dL. "
            "Tính giá trị trung bình và xác định số người theo ngưỡng tham chiếu."
        )
        self.assertIn("Trung bình: 127,6 mg/dL", answer)
        self.assertIn("bình thường: 1 (95 mg/dL)", answer)
        self.assertIn("tiền tiểu đường: 1 (110 mg/dL)", answer)
        self.assertIn("trong ngưỡng đái tháo đường: 3 (128, 145, 160 mg/dL)", answer)

    def test_supervisor_answers_inline_lab_threshold_question_without_irrelevant_retrieval(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm()
        result = Supervisor(llm, RetrievalService(rag)).process(
            "Tôi 25 tuổi, glucose lúc đói là 132 mg/dL sau 8 tiếng nhịn ăn. "
            "Có bình thường không, có nguy cơ tiểu đường không?"
        )
        self.assertEqual(result.metadata["provider"], "clinical_threshold_rule")
        self.assertIn("132 mg/dL thuộc trong ngưỡng đái tháo đường", result.answer)
        self.assertEqual(rag.calls, [])
        self.assertEqual(llm.calls, [])

    def test_supervisor_routes_inline_glucose_series_to_local_analysis(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm()
        result = Supervisor(llm, RetrievalService(rag)).process(
            "Cho bộ dữ liệu đường huyết lúc đói lần lượt là 95, 110, 128, 145 và 160 mg/dL. "
            "Tính giá trị trung bình, phân loại theo ngưỡng và nhận xét phân bố."
        )
        self.assertEqual(result.metadata["provider"], "local_analysis")
        self.assertIn("Trung bình: 127,6 mg/dL", result.answer)
        self.assertIn("tiền tiểu đường: 1 (110 mg/dL)", result.answer)
        self.assertEqual(rag.calls, [])
        self.assertEqual(llm.calls, [])

    def test_retrieval_and_answer_gates_reject_unrelated_mars_claims(self):
        rag = RetrievalService(FakeRag([diabetes_chunk()]))
        context = rag.search("Bệnh tiểu đường là gì?")
        self.assertFalse(context_supports_question("Bệnh tiểu đường ảnh hưởng thế nào đến định cư trên Sao Hỏa?", context))
        self.assertFalse(answer_supported_by_context(
            "Trên Sao Hỏa nhiệt độ thấp và vi khuẩn ảnh hưởng đến định cư.", context
        ))

    def test_answer_gate_checks_each_list_item_for_evidence(self):
        context = RetrievalService(FakeRag([diabetes_chunk()])).search("Bệnh tiểu đường là gì?")
        self.assertFalse(answer_supported_by_context(
            "Một số điểm:\n- Dinh dưỡng cân bằng.\n- Sao Hỏa có băng và vi khuẩn.", context
        ))

    def test_gi_gl_answer_must_cover_both_requested_measures(self):
        llm = FakeLlm("Chỉ số GI là thước đo tốc độ tăng glucose trong máu.")
        fallback = Mock(return_value="GI là chỉ số phản ánh mức tăng glucose.")
        chunk = diabetes_chunk()
        chunk["content"] += " Chỉ số GI phản ánh tốc độ tăng đường huyết; GL là tải đường huyết."
        result = Supervisor(llm, RetrievalService(FakeRag([chunk])), fallback).process(
            "Chỉ số GI và GL là gì?"
        )
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.answer, INSUFFICIENT)

    def test_repetition_filter_rejects_repeated_phrase(self):
        self.assertTrue(is_repetitive_or_corrupt(" ".join(["bí ngòi bắp đít"] * 8)))
        self.assertTrue(is_repetitive_or_corrupt("bí ngòi bắp đít rồi bí ngòi bắp đít"))
        self.assertFalse(is_repetitive_or_corrupt("Người bệnh nên ăn đa dạng và theo dõi khẩu phần."))

    def test_retrieval_adapter_calls_existing_search_once_and_keeps_internal_ids(self):
        rag = FakeRag([diabetes_chunk()])
        service = RetrievalService(rag)
        context = service.search("Bệnh tiểu đường là gì?")
        self.assertEqual(len(rag.calls), 1)
        self.assertEqual(context[0].source_id, "chunk:7")
        self.assertEqual(context[0].url, "https://internal.example/article")

    def test_medical_question_routes_to_one_agent_with_retrieved_context(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm()
        supervisor = Supervisor(llm, RetrievalService(rag))
        result = supervisor.process("Bệnh tiểu đường là gì?")
        self.assertEqual(result.agents_used, ["medical"])
        self.assertEqual(len(rag.calls), 1)
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("chunk:7", llm.calls[0][1])
        self.assertEqual(result.status, "completed")
        self.assertNotIn("https://", result.answer)

    def test_combined_question_uses_two_agents_and_one_shared_retrieval(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm("Bệnh tiểu đường và dinh dưỡng đều cần được lưu ý.")
        result = Supervisor(llm, RetrievalService(rag)).process(
            "Bệnh tiểu đường có biến chứng gì và người bệnh nên ăn gì?"
        )
        self.assertEqual(result.agents_used, ["medical", "nutrition"])
        self.assertEqual(len(rag.calls), 1)
        self.assertEqual(len(llm.calls), 2)

    def test_nutrition_question_routes_only_to_nutrition(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm()
        result = Supervisor(llm, RetrievalService(rag)).process(
            "Người mắc tiểu đường nên lưu ý gì về dinh dưỡng?"
        )
        self.assertEqual(result.agents_used, [])
        self.assertEqual(len(llm.calls), 1)

    def test_type_comparison_uses_one_medical_agent(self):
        chunk = diabetes_chunk()
        chunk["content"] = (
            "Tiểu đường type 1 và type 2 khác nhau về cơ chế. "
            "Type 1 thường do tự miễn làm giảm sản xuất insulin; "
            "type 2 thường liên quan đến đề kháng insulin."
        )
        rag = FakeRag([chunk])
        llm = FakeLlm("Type 1 và type 2 khác nhau về cơ chế.")
        result = Supervisor(llm, RetrievalService(rag)).process(
            "Tiểu đường type 1 và type 2 khác nhau thế nào?"
        )
        self.assertEqual(result.agents_used, [])
        self.assertEqual(len(rag.calls), 1)
        self.assertEqual(len(llm.calls), 1)

    def test_banana_and_gi_questions_route_to_nutrition(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm()
        supervisor = Supervisor(llm, RetrievalService(rag))
        self.assertEqual(supervisor._plan("Người mắc tiểu đường có ăn được chuối không?", "")["agents"], ["nutrition"])
        self.assertEqual(supervisor._plan("Chỉ số GI và GL là gì?", "")["agents"], ["nutrition"])

    def test_out_of_scope_skips_retrieval_and_llm(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm()
        result = Supervisor(llm, RetrievalService(rag)).process("Giá vàng hôm nay thế nào?")
        self.assertEqual(result.agents_used, [])
        self.assertEqual(rag.calls, [])
        self.assertEqual(llm.calls, [])

    def test_empty_retrieval_abstains_without_calling_llm(self):
        rag = FakeRag([])
        llm = FakeLlm()
        result = Supervisor(llm, RetrievalService(rag)).process("Bệnh tiểu đường là gì?")
        self.assertEqual(result.answer, INSUFFICIENT)
        self.assertEqual(llm.calls, [])
        self.assertEqual(result.status, "insufficient")

    def test_ollama_failure_uses_rule_fallback(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm(error=OllamaUnavailable("offline"))
        fallback = Mock(return_value="Câu trích xuất an toàn từ tài liệu.")
        result = Supervisor(llm, RetrievalService(rag), fallback).process(
            "Bệnh tiểu đường là gì?"
        )
        fallback.assert_called_once()
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.metadata["provider"], "rule_fallback")
        self.assertNotIn("offline", result.answer)

    def test_chinese_output_is_rejected_and_falls_back(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm("糖尿病是一种疾病")
        fallback = Mock(return_value="Tài liệu chưa đủ thông tin.")
        result = Supervisor(llm, RetrievalService(rag), fallback).process(
            "Bệnh tiểu đường là gì?"
        )
        fallback.assert_called_once()
        self.assertNotIn("糖尿病", result.answer)

    def test_non_vietnamese_output_is_rejected_and_falls_back(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm("Diabetes is a chronic metabolic disease affecting blood glucose levels.")
        fallback = Mock(return_value="Tài liệu chưa đủ thông tin.")
        result = Supervisor(llm, RetrievalService(rag), fallback).process(
            "Bệnh tiểu đường là gì?"
        )
        fallback.assert_called_once()
        self.assertNotIn("Diabetes is", result.answer)
        self.assertTrue(is_non_vietnamese("A short English answer about diabetes."))

    def test_answer_without_retrieved_term_support_falls_back(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm("Bệnh thận mạn tính luôn có thể chữa khỏi.")
        fallback = Mock(return_value="Tài liệu chưa đủ thông tin.")
        result = Supervisor(llm, RetrievalService(rag), fallback).process(
            "Bệnh tiểu đường là gì?"
        )
        fallback.assert_called_once()
        self.assertNotIn("thận mạn", result.answer)

    def test_numeric_table_summary_is_computed_from_input(self):
        summary = summarize_numeric_table("Ngày,Đường huyết\n1,5.0\n2,7.0\n3,6.0")
        self.assertIsNotNone(summary)
        self.assertEqual(summary["Đường huyết"]["count"], 3)
        self.assertEqual(summary["Đường huyết"]["mean"], 6.0)

    def test_data_agent_uses_computed_statistics_not_model_numbers(self):
        rag = FakeRag()
        llm = FakeLlm("The mean is 999 and the data proves a diagnosis.")
        result = Supervisor(llm, RetrievalService(rag)).process(
            "Phân tích dữ liệu này", data_text="Ngày,Đường huyết\n1,5\n2,7\n3,6"
        )
        self.assertEqual(result.agents_used, ["data_analysis"])
        self.assertIn("trung bình 6.0", result.answer)
        self.assertNotIn("999", result.answer)
        self.assertEqual(rag.calls, [])

    def test_personal_diagnosis_and_dose_requests_are_blocked_without_llm(self):
        rag = FakeRag([diabetes_chunk()])
        llm = FakeLlm()
        supervisor = Supervisor(llm, RetrievalService(rag))
        diagnosis = supervisor.process("Tôi có bị tiểu đường không?")
        dosage = supervisor.process("Tôi nên uống bao nhiêu thuốc metformin?")
        follow_up = supervisor.process(
            "Vậy tôi có bị không?", history=[{"content": "Tôi có chỉ số đường huyết cao."}]
        )
        self.assertEqual(diagnosis.metadata["provider"], "safety_rule")
        self.assertEqual(dosage.metadata["provider"], "safety_rule")
        self.assertEqual(follow_up.metadata["provider"], "safety_rule")
        self.assertEqual(llm.calls, [])
        self.assertEqual(rag.calls, [])

    def test_answer_filter_removes_links_citations_and_source_ids(self):
        answer = _clean_answer("Đáp án [1] [chunk:12] source_id=abc https://a.test/x")
        self.assertNotIn("[1]", answer)
        self.assertNotIn("chunk:12", answer)
        self.assertNotIn("source_id", answer)
        self.assertNotIn("http", answer)


class ConversationStoreTests(unittest.TestCase):
    def test_metadata_is_saved_but_not_exposed_in_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ConversationStore(Path(temp_dir) / "history.sqlite3")
            conversation = store.create_conversation()
            store.add_message(conversation["id"], "user", "Câu hỏi")
            store.add_message(
                conversation["id"], "assistant", "Trả lời", agents_used=["medical"],
                metadata={"source_ids": ["chunk:9"], "source_metadata": [{"url": "secret"}]},
            )
            history = store.get_messages(conversation["id"])
            self.assertNotIn("metadata", history[1])
            self.assertNotIn("chunk:9", json.dumps(history, ensure_ascii=False))
            self.assertTrue(store.delete_conversation(conversation["id"]))
            self.assertEqual(store.list_conversations(), [])


class AttachmentExtractionTests(unittest.TestCase):
    def test_txt_and_csv_extraction(self):
        self.assertEqual(_extract(".txt", "xin chào".encode())[0], "xin chào")
        self.assertIn("glucose", _extract(".csv", b"day,glucose\n1,5\n2,7")[0])

    def test_docx_extraction_uses_document_xml(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<w:document xmlns:w='urn:w'><w:body><w:p><w:r><w:t>Diabetes note</w:t></w:r></w:p></w:body></w:document>")
        text, status = _extract(".docx", buffer.getvalue())
        self.assertEqual(status, "extracted")
        self.assertEqual(text, "Diabetes note")

    def test_pdf_text_and_scan_are_distinguished(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        buffer = io.BytesIO()
        writer = PdfWriter()
        page = writer.add_blank_page(width=100, height=100)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 80 Td (Diabetes PDF text) Tj ET")
        page[NameObject("/Contents")] = stream
        writer.write(buffer)
        text, status = _extract(".pdf", buffer.getvalue())
        self.assertIn("Diabetes PDF text", text)
        self.assertEqual(status, "extracted")
        scan = io.BytesIO()
        scan_writer = PdfWriter()
        scan_writer.add_blank_page(width=100, height=100)
        scan_writer.write(scan)
        self.assertEqual(_extract(".pdf", scan.getvalue())[1], "pdf_no_text")

    def test_image_is_never_assumed_readable_without_ocr(self):
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j3S8AAAAASUVORK5CYII=")
        _text, status = _extract(".png", png)
        self.assertIn(status, {"ocr_extracted", "ocr_unavailable", "ocr_languages_unavailable", "ocr_no_text", "ocr_failed"})

    def test_upload_rejects_unsupported_and_oversized_files(self):
        from werkzeug.datastructures import FileStorage
        service = AttachmentService(Path(tempfile.gettempdir()) / "hybridrag-attachment-tests")
        with self.assertRaises(AttachmentError):
            service.save("conversation", [FileStorage(stream=io.BytesIO(b"x"), filename="payload.exe")])
        with self.assertRaises(AttachmentError):
            service.save("conversation", [FileStorage(stream=io.BytesIO(b"x" * (8 * 1024 * 1024 + 1)), filename="large.txt")])


class FlaskIntegrationTests(unittest.TestCase):
    def setUp(self):
        import app as app_module
        self.app_module = app_module
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.temp_dir.name) / "chat.sqlite3")
        self.store_patch = patch.object(app_module, "conversation_store", self.store)
        self.store_patch.start()
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        self.store_patch.stop()
        self.temp_dir.cleanup()

    def test_legacy_ask_endpoint_remains_available(self):
        fake_rag = Mock()
        fake_rag.ask.return_value = {"answer": "legacy", "sources": [], "retrieved_count": 0}
        with patch.object(self.app_module, "get_rag", return_value=fake_rag):
            response = self.client.post("/api/ask", json={"question": "test"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["answer"], "legacy")

    def test_health_and_conversation_delete(self):
        with patch.object(self.app_module._ollama, "health", return_value={
            "server": "unavailable", "model_installed": False,
            "model_loaded": False, "model_state": "server_unavailable",
            "waiting_requests": 0, "error": "server_connection_error",
        }):
            health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json["ollama"], "unavailable")
        created = self.client.post("/api/conversations", json={}).json["conversation"]
        deleted = self.client.delete(f"/api/conversations/{created['id']}")
        self.assertEqual(deleted.status_code, 200)

    def test_attachment_endpoints_are_removed(self):
        conversation_id = self.client.post("/api/conversations", json={}).json["conversation"]["id"]
        response = self.client.post(
            f"/api/conversations/{conversation_id}/attachments",
            data={"files": (io.BytesIO(b"not an image"), "fake.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get("/api/attachments/nonexistent").status_code, 404)

    def test_risk_estimate_endpoint_validates_inputs_and_returns_model_result(self):
        with self.assertRaises(RiskInputError):
            validate_input({"age": 999})
        sample = {
            "gender": "Female", "age": 54, "hypertension": "0", "heart_disease": "0",
            "smoking_history": "never", "bmi": 27.3, "HbA1c_level": 5.7,
            "blood_glucose_level": 126,
        }
        expected = {
            "risk_level": "high", "risk_label": "Nguy cơ cao",
            "status": "experimental_estimate", "message": "dataset estimate", "clinical_note": "",
        }
        with patch.object(self.app_module, "predict_risk", return_value=expected):
            response = self.client.post("/api/predict-risk", json=sample)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["risk_level"], "high")
        self.assertEqual(response.json["status"], "experimental_estimate")

    def test_risk_category_honors_lab_threshold_and_avoids_claiming_zero_risk(self):
        class FixedModel:
            def predict_proba(self, _frame):
                return [[0.99, 0.01]]

        values = {
            "gender": "Female", "age": 54, "hypertension": 0, "heart_disease": 0,
            "smoking_history": "never", "bmi": 27.3, "HbA1c_level": 5.7,
            "blood_glucose_level": 126, "fasting_status": "yes",
        }
        with patch("services.diabetes_risk._get_model", return_value=(FixedModel(), {})):
            result = predict_risk(values)
        self.assertEqual(result["risk_level"], "high")
        self.assertEqual(result["risk_label"], "Nguy cơ cao")
        self.assertIn("ngưỡng xét nghiệm", result["clinical_note"])

        values["fasting_status"] = "unknown"
        with patch("services.diabetes_risk._get_model", return_value=(FixedModel(), {})):
            result = predict_risk(values)
        self.assertEqual(result["risk_level"], "not_high")
        self.assertEqual(result["risk_label"], "Không thuộc nhóm nguy cơ cao")
        self.assertIn("không có nghĩa là hoàn toàn không có nguy cơ", result["message"])

    def test_chat_sse_and_history_hide_internal_metadata(self):
        def fake_process(question, history=None, data_text="", emit=None):
            emit("status", {"agent": "medical", "status": "running", "message": "Medical đang xử lý"})
            return ChatResponse(
                "Câu trả lời ngắn.", ["medical"], "completed",
                {"source_ids": ["chunk:44"], "source_metadata": [{"url": "private"}]},
            )
        with patch.object(self.app_module.supervisor, "process", side_effect=fake_process):
            response = self.client.post("/api/chat", json={"question": "Bệnh tiểu đường là gì?"})
            body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: status", body)
        self.assertIn("event: final", body)
        self.assertIn("Câu trả lời ngắn.", body)
        self.assertNotIn("chunk:44", body)
        conversation_id = json.loads(next(line[6:] for line in body.splitlines() if line.startswith("data: ") and "conversation_id" in line))
        history_response = self.client.get(f"/api/conversations/{conversation_id['conversation_id']}")
        self.assertEqual(history_response.status_code, 200)
        self.assertNotIn("metadata", history_response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
