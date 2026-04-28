"""
Bot logic: business hours, human-like delays, typing simulation, escalation.
"""

import os
import asyncio
import random
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional
import httpx

import database as db
import rag_engine

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TIMEZONE = os.getenv("TIMEZONE", "America/Argentina/Buenos_Aires")
BUSINESS_START = int(os.getenv("BUSINESS_START", "9"))
BUSINESS_END = int(os.getenv("BUSINESS_END", "18"))

TYPING_ACTION_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendChatAction" if TELEGRAM_TOKEN else None

# Smalltalk detection to save LLM tokens on greetings
SMALLTALK_PATTERNS = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "hey", "saludos",
    "gracias", "muchas gracias", "ok", "okay", "perfecto", "entendido", "dale",
    "cómo estás", "qué tal", "todo bien"
]


def _normalize(text: str) -> str:
    return text.lower().strip().rstrip(".?!")


def is_smalltalk(text: str) -> bool:
    t = _normalize(text)
    return any(p in t for p in SMALLTALK_PATTERNS) and len(t) < 60


def is_business_hours() -> bool:
    """Check if current time is within configured business hours, Mon-Fri."""
    now = datetime.now(ZoneInfo(TIMEZONE))
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return BUSINESS_START <= now.hour < BUSINESS_END


async def send_typing_action(chat_id: int) -> None:
    """Send 'typing' chat action to Telegram."""
    if not TYPING_ACTION_URL:
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(TYPING_ACTION_URL, json={
                "chat_id": chat_id,
                "action": "typing"
            }, timeout=10.0)
    except Exception as e:
        print(f"[Bot] Failed to send typing action: {e}")


async def simulate_human_delay(base: float = 2.5, variation: float = 2.5) -> None:
    """Sleep for a human-like random delay."""
    delay = base + random.random() * variation
    await asyncio.sleep(delay)


def get_out_of_hours_message() -> str:
    return (
        "Hola. El equipo de Auditoría Interna atiende consultas de lunes a viernes "
        f"de {BUSINESS_START}:00 a {BUSINESS_END}:00 hs (horario local). \n\n"
        "Tu mensaje quedó registrado y será respondido en el próximo horario laboral. "
        "Si la consulta es urgente, por favor contactá directamente a tu supervisor."
    )


def get_smalltalk_response(text: str) -> str:
    t = _normalize(text)
    if any(w in t for w in ["gracias", "muchas gracias"]):
        return "De nada. Quedo a disposición para cualquier otra consulta de Auditoría."
    if any(w in t for w in ["ok", "okay", "perfecto", "entendido", "dale"]):
        return "Perfecto. Avísame si necesitás algo más."
    return (
        "Hola, soy del equipo de Auditoría Interna. "
        "¿En qué puedo ayudarte hoy? Podés consultarme sobre normativas, procedimientos o casos concretos."
    )


async def process_message(
    chat_id: int,
    text: str,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main message processing pipeline.
    Returns dict with: reply_text, confidence, escalated, error
    """
    # 1. Persist user
    db.upsert_user(chat_id, username, first_name, last_name)

    # 2. Out of hours check
    if not is_business_hours():
        return {
            "reply_text": get_out_of_hours_message(),
            "confidence": 1.0,
            "escalated": False,
            "error": None
        }

    # 3. Smalltalk shortcut
    if is_smalltalk(text):
        return {
            "reply_text": get_smalltalk_response(text),
            "confidence": 1.0,
            "escalated": False,
            "error": None
        }

    # 4. Typing indicator + human delay
    await send_typing_action(chat_id)
    await simulate_human_delay()

    # 5. Retrieve recent history for context
    history = db.get_conversation_history(chat_id, limit=3)

    # 6. RAG query
    try:
        rag_result = rag_engine.query_rag(text, chat_history=history)
    except Exception as e:
        print(f"[Bot] RAG error: {e}")
        return {
            "reply_text": (
                "Disculpá, estoy teniendo dificultades técnicas para acceder a la documentación en este momento. "
                "Por favor, intentá de nuevo en unos minutos o contactá directamente al equipo de Auditoría."
            ),
            "confidence": 0.0,
            "escalated": True,
            "error": str(e)
        }

    # 7. Persist conversation
    conv_id = db.save_conversation(
        chat_id=chat_id,
        user_query=text,
        bot_response=rag_result["response"],
        confidence_score=rag_result["confidence"],
        was_escalated=rag_result["escalated"]
    )

    # 8. Persist document references
    if rag_result.get("citations"):
        db.save_document_refs(conv_id, rag_result["citations"])

    # 9. Create escalation ticket if needed
    if rag_result["escalated"]:
        db.create_escalation(
            chat_id=chat_id,
            query=text,
            context_summary=rag_result["response"][:500],
            reason=f"Confianza baja ({rag_result['confidence']})"
        )

    return {
        "reply_text": rag_result["response"],
        "confidence": rag_result["confidence"],
        "escalated": rag_result["escalated"],
        "error": None,
        "model_used": rag_result.get("model_used")
    }
