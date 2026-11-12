import os
import queue
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from time import perf_counter

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from agents.supervisor import Supervisor
from schemas.messages import ChatResponse
from services.conversation_store import ConversationStore
from services.attachment_service import AttachmentService
from services.diabetes_risk import RiskInputError, predict_risk
from services.event_stream import encode_sse
from services.ollama_client import OllamaClient
from services.retrieval_service import RetrievalService

app = Flask(__name__)

_rag = None
_rag_lock = threading.Lock()
_ollama = OllamaClient()
_database_path = Path(os.getenv("CHAT_DATABASE", str(BASE_DIR / "data" / "chat_history.sqlite3")))
conversation_store = ConversationStore(_database_path)
# Retain cleanup for files attached by older conversations; no upload or fetch route is exposed.
attachment_service = AttachmentService(BASE_DIR / "data" / "chat_uploads")


class _LazyRAG:
    def search(self, question, top_k=5):
        return get_rag().search(question, top_k=top_k)


def get_rag():
    global _rag
    if _rag is None:
        with _rag_lock:
            if _rag is None:
                # Query embeddings must use the already cached model; never fetch weights at runtime.
                os.environ["HF_HUB_OFFLINE"] = "1"
                from hybrid_search import HybridRAG
                _rag = HybridRAG()
    return _rag


supervisor = Supervisor(_ollama, RetrievalService(_LazyRAG()))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    ollama_state = _ollama.health()
    return jsonify({
        "status": "ok",
        "message": "Diabetes Research AI đang hoạt động",
        # Keep the old field for existing consumers; also expose the distinction.
        "ollama": "available" if (
            ollama_state["server"] == "available" and ollama_state["model_installed"]
        ) else "unavailable",
        "ollama_server": ollama_state["server"],
        "model": _ollama.model,
        "model_installed": ollama_state["model_installed"],
        "model_loaded": ollama_state["model_loaded"],
        "model_state": ollama_state["model_state"],
        "waiting_requests": ollama_state["waiting_requests"],
        "ollama_error": ollama_state["error"],
        "retrieval": "initialized" if _rag is not None else "lazy",
    })


@app.route("/api/predict-risk", methods=["POST"])
def predict_diabetes_risk():
    data = request.get_json(silent=True)
    try:
        result = predict_risk(data)
        return jsonify(result)
    except RiskInputError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError:
        app.logger.exception("Diabetes risk dataset is unavailable")
        return jsonify({"error": "Dataset dự đoán hiện chưa sẵn sàng."}), 503
    except Exception:
        app.logger.exception("Dataset risk model failed")
        return jsonify({"error": "Chưa thể tính ước tính từ dataset lúc này."}), 503


@app.route("/api/ask", methods=["POST"])
def ask():
    """Backward-compatible endpoint; keeps the original HybridRAG response."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    question = data.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Vui lòng nhập câu hỏi."}), 400
    domain = data.get("domain", "diabetes")
    if not isinstance(domain, str) or domain.strip().lower() != "diabetes":
        return jsonify({"error": "Lĩnh vực không hợp lệ."}), 400
    try:
        return jsonify(get_rag().ask(question.strip(), domain="diabetes"))
    except Exception:
        app.logger.exception("Lỗi khi xử lý API /api/ask")
        return jsonify({"error": "Không thể xử lý câu hỏi. Hãy kiểm tra dịch vụ retrieval."}), 500


@app.route("/api/conversations", methods=["GET", "POST"])
def conversations():
    if request.method == "GET":
        return jsonify({"conversations": conversation_store.list_conversations()})
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    title = data.get("title", "Cuộc trò chuyện mới")
    if not isinstance(title, str):
        title = "Cuộc trò chuyện mới"
    return jsonify({"conversation": conversation_store.create_conversation(title)}), 201


@app.route("/api/conversations/<conversation_id>", methods=["GET", "DELETE"])
def conversation(conversation_id):
    if request.method == "DELETE":
        if not conversation_store.delete_conversation(conversation_id):
            return jsonify({"error": "Không tìm thấy cuộc trò chuyện."}), 404
        attachment_service.delete_conversation(conversation_id)
        return jsonify({"status": "deleted"})
    if not conversation_store.exists(conversation_id):
        return jsonify({"error": "Không tìm thấy cuộc trò chuyện."}), 404
    return jsonify({
        "conversation_id": conversation_id,
        "messages": conversation_store.get_messages(conversation_id),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    question = data.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Vui lòng nhập câu hỏi."}), 400
    question = question.strip()
    if len(question) > 12000:
        return jsonify({"error": "Câu hỏi vượt quá giới hạn 12.000 ký tự."}), 413

    conversation_id = data.get("conversation_id")
    if conversation_id:
        if not isinstance(conversation_id, str) or not conversation_store.exists(conversation_id):
            return jsonify({"error": "Không tìm thấy cuộc trò chuyện."}), 404
    else:
        conversation_id = conversation_store.create_conversation(question[:70])["id"]

    history = conversation_store.get_messages(conversation_id)[-8:]
    data_text = data.get("data", "")
    if not isinstance(data_text, str):
        data_text = ""
    if not data_text:
        data_text = _extract_table(question)
    user_message = conversation_store.add_message(conversation_id, "user", question)
    events = queue.Queue()
    stop_marker = object()

    def work():
        request_started = perf_counter()
        try:
            result = supervisor.process(
                question,
                history=history,
                data_text=data_text,
                emit=lambda event_name, payload: events.put((event_name, payload)),
            )
            elapsed_ms = round((perf_counter() - request_started) * 1000)
            result.metadata["timings_ms"] = result.metadata.get("timings_ms", {})
            result.metadata["timings_ms"]["total"] = elapsed_ms
            assistant_message = conversation_store.add_message(
                conversation_id,
                "assistant",
                result.answer,
                status=result.status,
                agents_used=result.agents_used,
                metadata=result.metadata,
            )
            app.logger.info("chat complete conversation=%s status=%s agents=%s total_ms=%d stages=%s",
                            conversation_id, result.status, result.agents_used, elapsed_ms,
                            result.metadata.get("timings_ms", {}))
            payload = result.public_dict()
            payload.update({
                "conversation_id": conversation_id,
                "message_id": assistant_message["id"],
            })
            events.put(("final", payload))
        except Exception:
            app.logger.exception("Lỗi không xử lý được trong chat")
            message = "Hệ thống gặp lỗi khi xử lý. Bạn có thể thử lại sau."
            try:
                assistant_message = conversation_store.add_message(
                    conversation_id, "assistant", message, status="error"
                )
                payload = {
                    "answer": message,
                    "agents_used": [],
                    "status": "error",
                    "conversation_id": conversation_id,
                    "message_id": assistant_message["id"],
                }
                events.put(("final", payload))
            except Exception:
                events.put(("error", {"message": message}))
        finally:
            events.put(("done", {}))
            events.put((stop_marker, None))

    def generate():
        yield encode_sse("request_started", {
            "conversation_id": conversation_id,
            "message_id": user_message["id"],
        })
        threading.Thread(target=work, daemon=True).start()
        while True:
            event_name, payload = events.get()
            if event_name is stop_marker:
                break
            yield encode_sse(event_name, payload)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _extract_table(text):
    fenced = re.search(r"```(?:csv|tsv|text)?\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    lines = [line for line in text.splitlines() if "," in line or "\t" in line or ";" in line]
    return "\n".join(lines) if len(lines) >= 3 else ""


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=5035, debug=False, threaded=True)
    finally:
        if _rag is not None:
            _rag.close()
