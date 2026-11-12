from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    source_id: str
    chunk_id: str
    title: str
    url: str
    content: str
    score: float | None = None


@dataclass
class AgentResult:
    agent: str
    status: str
    answer: str = ""
    error: str | None = None
    source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChatResponse:
    answer: str
    agents_used: list[str]
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "agents_used": self.agents_used,
            "status": self.status,
        }
