"""
RAG Engine: retrieves relevant context from ChromaDB and queries Groq LLM.
"""

import os
from typing import List, Dict, Any, Optional
from groq import Groq
from embedder import buscar

# Groq config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-70b-versatile")
MAX_TOKENS = 2048
TEMPERATURE = 0.3
CONFIDENCE_THRESHOLD = 0.40


def _get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)


def query_rag(user_query: str, chat_history=None):
    """
    Main RAG pipeline.
    Returns: {response, confidence, citations, escalated}
    """
    # 1. Retrieve context
    documentos = buscar(user_query, n_resultados=3)
    
    if not documentos:
        return {
            "response": "No encontré información relevante en la documentación. Te sugiero consultar con el Jefe de Auditoría.",
            "confidence": 0.0,
            "citations": [],
            "escalated": True
        }
    
    # 2. Build context
    context_parts = []
    citations = []
    for i, doc in enumerate(documentos, 1):
        context_parts.append(f"[Documento {i}] {doc['fuente']}:\n{doc['texto'][:600]}")
        citations.append({
            "doc": doc['fuente'],
            "chunk": i,
            "score": doc['relevancia']
        })
    
    context_str = "\n\n---\n\n".join(context_parts)
    
    # 3. Compute confidence
    confidence = sum(d['relevancia'] for d in documentos) / len(documentos)
    
    # 4. Build messages
    system_prompt = (
        "Sos un Auditor Interno Senior. Respondé basándote ÚNICAMENTE en la documentación proporcionada. "
        "Cita las fuentes como [Fuente: nombre_documento]. "
        "Si no tenés información suficiente, indicá que se debe escalar la consulta. "
        "NUNCA digas que sos una IA. Usá tono profesional y claro."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Contexto:\n\n{context_str}\n\n---\n\nConsulta: {user_query}"}
    ]
    
    # 5. Call Groq
    client = _get_groq_client()
    try:
        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    except Exception:
        chat_completion = client.chat.completions.create(
            model=GROQ_FALLBACK_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    
    response_text = chat_completion.choices[0].message.content.strip()
    
    # 6. Escalation
    escalated = confidence < CONFIDENCE_THRESHOLD
    if escalated:
        response_text += "\n\n⚠️ Esta consulta requiere revisión adicional del equipo de Auditoría."
    
    return {
        "response": response_text,
        "confidence": confidence,
        "citations": citations,
        "escalated": escalated,
        "model_used": chat_completion.model
    }


def quick_check_index() -> bool:
    """Check if the Chroma collection has data."""
    try:
        from embedder import get_or_create_collection
        collection = get_or_create_collection()
        return collection.count() > 0
    except Exception:
        return False
