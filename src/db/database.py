"""
AIGText — Conversation History Database
SQLite-based persistent storage for chat conversations.
"""
import sqlite3
import json
import os
import uuid
import threading
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))

_db_path = None
_local = threading.local()


def init_db(db_path="data/conversations.db"):
    """Initialize database, create tables if not exist."""
    global _db_path
    _db_path = db_path
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = _connect()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT '新对话',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content         TEXT NOT NULL,
            image_data      TEXT,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
    """)
    conn.commit()
    conn.close()


def _connect():
    """Get thread-local connection."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_db_path or "data/conversations.db")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


def _now():
    return datetime.now(CST).isoformat(timespec="seconds")


# ── CRUD ──

def create_conversation(title="新对话"):
    cid = str(uuid.uuid4())[:8]
    now = _now()
    _connect().execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (cid, title, now, now)
    ).connection.commit()
    return {"id": cid, "title": title, "created_at": now, "updated_at": now}


def list_conversations():
    rows = _connect().execute("""
        SELECT c.id, c.title, c.created_at, c.updated_at,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
        FROM conversations c
        ORDER BY c.updated_at DESC
    """).fetchall()
    return [dict(r) for r in rows]


def get_conversation(cid):
    conv = _connect().execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
    if not conv:
        return None
    messages = _connect().execute(
        "SELECT id, role, content, image_data, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
        (cid,)
    ).fetchall()
    result_messages = []
    for m in messages:
        d = dict(m)
        if d.get("image_data") and isinstance(d["image_data"], str):
            try:
                d["image_data"] = json.loads(d["image_data"])
            except (json.JSONDecodeError, TypeError):
                pass  # 保持原始字符串（旧数据或单图 URL）
        result_messages.append(d)
    return {"id": conv["id"], "title": conv["title"],
            "created_at": conv["created_at"], "updated_at": conv["updated_at"],
            "messages": result_messages}


def delete_conversation(cid):
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
    conn.commit()


def update_conversation_title(cid, title):
    now = _now()
    _connect().execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (title, now, cid)
    ).connection.commit()


def save_messages(cid, messages):
    """Replace all messages for a conversation. `messages` is a list of {role, content, image_data?}."""
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
    now = _now()
    for msg in messages:
        img_data = msg.get("image_data")
        if isinstance(img_data, list):
            img_data = json.dumps(img_data, ensure_ascii=False)
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, image_data, created_at) VALUES (?, ?, ?, ?, ?)",
             (cid, msg.get("role", "user"), msg.get("content", ""),
              img_data, now)
        )
    # Auto-title: use first user message (first 30 chars)
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user:
        raw_title = first_user.get("content", "").strip()
        title = raw_title[:30] + ("..." if len(raw_title) > 30 else "")
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, cid)
        )
    else:
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
    conn.commit()


def get_all_messages():
    """Return all messages from all conversations in a single query.
    Used by learning/ page to avoid N+1 HTTP requests.
    Returns list of {id, role, content, image_data, created_at, conversation_id}.
    """
    messages = _connect().execute("""
        SELECT m.id, m.role, m.content, m.image_data, m.created_at,
               m.conversation_id, c.title AS conversation_title
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        ORDER BY m.id
    """).fetchall()
    result = []
    for m in messages:
        d = dict(m)
        if d.get("image_data") and isinstance(d["image_data"], str):
            try:
                d["image_data"] = json.loads(d["image_data"])
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(d)
    return result
