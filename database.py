"""
Database module for Internal Audit Chatbot.
SQLite for conversation history, user profiles, and escalation tracking.
"""

import sqlite3
import json
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import os

DB_PATH = os.getenv("DB_PATH", "./data/auditoria.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    with _get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                chat_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                department TEXT DEFAULT 'Sin departamento',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                user_query TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                confidence_score REAL,
                was_escalated INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES users(chat_id)
            );

            CREATE TABLE IF NOT EXISTS escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                context_summary TEXT,
                reason TEXT DEFAULT 'Confianza baja',
                status TEXT DEFAULT 'PENDIENTE',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES users(chat_id)
            );

            CREATE TABLE IF NOT EXISTS document_refs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                document_name TEXT NOT NULL,
                chunk_index INTEGER,
                similarity_score REAL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_conv_chat ON conversations(chat_id);
            CREATE INDEX IF NOT EXISTS idx_esc_status ON escalations(status);
        """)
        conn.commit()


# -------------------- USERS --------------------

def upsert_user(chat_id: int, username: Optional[str], first_name: Optional[str],
                last_name: Optional[str]) -> None:
    with _lock, _get_connection() as conn:
        conn.execute("""
            INSERT INTO users (chat_id, username, first_name, last_name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                username=excluded.username,
                first_name=excluded.first_name,
                last_name=excluded.last_name,
                last_active=CURRENT_TIMESTAMP
        """, (chat_id, username, first_name, last_name))
        conn.commit()


def get_user(chat_id: int) -> Optional[Dict[str, Any]]:
    with _lock, _get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE chat_id = ?", (chat_id,)).fetchone()
        return dict(row) if row else None


# -------------------- CONVERSATIONS --------------------

def save_conversation(chat_id: int, user_query: str, bot_response: str,
                      confidence_score: float, was_escalated: bool = False) -> int:
    with _lock, _get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO conversations (chat_id, user_query, bot_response, confidence_score, was_escalated)
            VALUES (?, ?, ?, ?, ?)
        """, (chat_id, user_query, bot_response, confidence_score, int(was_escalated)))
        conn.commit()
        return cursor.lastrowid


def get_conversation_history(chat_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    with _lock, _get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM conversations
            WHERE chat_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (chat_id, limit)).fetchall()
        return [dict(r) for r in rows]


# -------------------- ESCALATIONS --------------------

def create_escalation(chat_id: int, query: str, context_summary: str = "",
                      reason: str = "Confianza baja") -> int:
    with _lock, _get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO escalations (chat_id, query, context_summary, reason)
            VALUES (?, ?, ?, ?)
        """, (chat_id, query, context_summary, reason))
        conn.commit()
        return cursor.lastrowid


def get_pending_escalations() -> List[Dict[str, Any]]:
    with _lock, _get_connection() as conn:
        rows = conn.execute("""
            SELECT e.*, u.username, u.first_name, u.last_name
            FROM escalations e
            LEFT JOIN users u ON e.chat_id = u.chat_id
            WHERE e.status = 'PENDIENTE'
            ORDER BY e.created_at ASC
        """).fetchall()
        return [dict(r) for r in rows]


# -------------------- DOCUMENT REFERENCES --------------------

def save_document_refs(conversation_id: int, refs: List[Dict[str, Any]]) -> None:
    with _lock, _get_connection() as conn:
        for ref in refs:
            conn.execute("""
                INSERT INTO document_refs (conversation_id, document_name, chunk_index, similarity_score)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, ref.get("doc"), ref.get("chunk"), ref.get("score")))
        conn.commit()


# -------------------- STATISTICS --------------------

def get_user_stats(chat_id: int) -> Dict[str, Any]:
    with _lock, _get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) as c FROM conversations WHERE chat_id = ?", (chat_id,)
        ).fetchone()["c"]
        escalated = conn.execute(
            "SELECT COUNT(*) as c FROM conversations WHERE chat_id = ? AND was_escalated = 1", (chat_id,)
        ).fetchone()["c"]
        avg_conf = conn.execute(
            "SELECT AVG(confidence_score) as avg FROM conversations WHERE chat_id = ?", (chat_id,)
        ).fetchone()["avg"] or 0.0
        return {"total_queries": total, "escalated": escalated, "avg_confidence": round(avg_conf, 2)}


def get_global_stats() -> Dict[str, Any]:
    with _lock, _get_connection() as conn:
        total_users = conn.execute("SELECT COUNT(DISTINCT chat_id) as c FROM users").fetchone()["c"]
        total_queries = conn.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]
        total_escalations = conn.execute(
            "SELECT COUNT(*) as c FROM escalations WHERE status = 'PENDIENTE'"
        ).fetchone()["c"]
        avg_conf = conn.execute(
            "SELECT AVG(confidence_score) as avg FROM conversations"
        ).fetchone()["avg"] or 0.0
        return {
            "total_users": total_users,
            "total_queries": total_queries,
            "pending_escalations": total_escalations,
            "avg_confidence": round(avg_conf, 2)
        }


# Initialize on import
init_db()
