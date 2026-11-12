from agents.base import BaseAgent


class KnowledgeAgent(BaseAgent):
    name = "knowledge"
    system_prompt = (
        "Bạn tổng hợp khái niệm và mối quan hệ về bệnh tiểu đường từ tài liệu "
        "được cung cấp. Trả lời tiếng Việt ngắn gọn. Không suy ra nhân quả chỉ vì "
        "hai khái niệm cùng xuất hiện. Phân biệt điều tài liệu nói với phần chưa "
        "đủ căn cứ. Không in URL, source ID, citation hay mục nguồn."
    )
