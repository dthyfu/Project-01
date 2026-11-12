import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    """SQLite persistence for chat messages and private processing metadata."""

    def __init__(self, database_path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    agents_used_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation_idx
                    ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
                    original_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    storage_name TEXT NOT NULL,
                    extracted_text TEXT NOT NULL DEFAULT '',
                    extraction_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS attachments_conversation_idx
                    ON attachments(conversation_id, created_at);
            """)

    def create_conversation(self, title="Cuộc trò chuyện mới"):
        conversation_id = str(uuid.uuid4())
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (conversation_id, title[:100], now, now),
            )
        return {"id": conversation_id, "title": title[:100], "created_at": now, "updated_at": now}

    def exists(self, conversation_id):
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone() is not None

    def add_message(self, conversation_id, role, content, status="completed", agents_used=None, metadata=None):
        message_id = str(uuid.uuid4())
        now = _now()
        agents_json = json.dumps(agents_used or [], ensure_ascii=False)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(id,conversation_id,role,content,status,agents_used_json,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (message_id, conversation_id, role, content, status, agents_json, metadata_json, now),
            )
            if role == "user":
                connection.execute(
                    "UPDATE conversations SET title=? WHERE id=? AND title='Cuộc trò chuyện mới' "
                    "AND (SELECT COUNT(*) FROM messages WHERE conversation_id=? AND role='user')=1",
                    (content.strip().splitlines()[0][:70] or "Cuộc trò chuyện mới", conversation_id, conversation_id),
                )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id)
            )
        return {"id": message_id, "role": role, "content": content, "status": status,
                "agents_used": agents_used or [], "created_at": now}

    def add_attachments(self, attachments):
        now = _now()
        with self._connect() as connection:
            for item in attachments:
                connection.execute(
                    """INSERT INTO attachments
                    (id,conversation_id,original_name,mime_type,size_bytes,storage_name,extracted_text,extraction_status,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (item["id"], item["conversation_id"], item["original_name"], item["mime_type"],
                     item["size_bytes"], item["storage_name"], item["extracted_text"], item["extraction_status"], now),
                )

    def get_attachments(self, attachment_ids, conversation_id):
        if not attachment_ids:
            return []
        placeholders = ",".join("?" for _ in attachment_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM attachments WHERE conversation_id=? AND id IN ({placeholders})",
                [conversation_id, *attachment_ids],
            ).fetchall()
        return [dict(row) for row in rows]

    def attach_to_message(self, attachment_ids, message_id, conversation_id):
        if not attachment_ids:
            return
        placeholders = ",".join("?" for _ in attachment_ids)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE attachments SET message_id=? WHERE conversation_id=? AND id IN ({placeholders})",
                [message_id, conversation_id, *attachment_ids],
            )

    def public_attachments(self, message_id):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,original_name,mime_type,size_bytes,extraction_status FROM attachments WHERE message_id=?",
                (message_id,),
            ).fetchall()
        return [{**dict(row), "url": f"/api/attachments/{row['id']}"} for row in rows]

    def get_attachment(self, attachment_id):
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM attachments WHERE id=?", (attachment_id,)).fetchone()
        return dict(row) if row else None

    def delete_attachment(self, attachment_id, conversation_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT storage_name FROM attachments WHERE id=? AND conversation_id=?",
                (attachment_id, conversation_id),
            ).fetchone()
            if not row:
                return None
            connection.execute("DELETE FROM attachments WHERE id=?", (attachment_id,))
        return row["storage_name"]

    def list_conversations(self):
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT c.id,c.title,c.created_at,c.updated_at,
                       (SELECT content FROM messages m WHERE m.conversation_id=c.id
                        ORDER BY m.created_at DESC LIMIT 1) AS preview
                FROM conversations c ORDER BY c.updated_at DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, conversation_id):
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT id,role,content,status,agents_used_json,created_at
                FROM messages WHERE conversation_id=? ORDER BY rowid
            """, (conversation_id,)).fetchall()
        return [{"id": row["id"], "role": row["role"], "content": row["content"],
                 "status": row["status"], "agents_used": json.loads(row["agents_used_json"]),
                 "created_at": row["created_at"]} for row in rows]

    def delete_conversation(self, conversation_id):
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            return cursor.rowcount > 0
