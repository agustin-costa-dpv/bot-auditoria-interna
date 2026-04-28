"""
RAG Engine: retrieves relevant context from ChromaDB and queries Groq LLM.
"""

import os
import json
from typing import List, Dict, Any, Optional
from groq import Groq
from embedder import buscar

# Groq config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-70b-versatile")
MAX_TOKENS = 2048
TEMPERATURE = 0.3

# RAG config
TOP_K = 5
MIN_DOCS = 1
CONFIDENCE_THRESHOLD = 0.40  # below this -> escalate

def _get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)


def _format_context(results: Dict[str, Any]) -> tuple:
    """Format retrieved chunks into a context string and citation list."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    context_parts = []
    citations = []
    seen_sources = set()
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        source = meta.get("source", "Desconocido")
        chunk_idx = meta.get("chunk_index", 0)
        total = meta.get("total_chunks", 1)
        
        # Deduplicate source mentions for cleaner citations
        if source not in seen_sources:
            seen_sources.add(source)
        
        context_parts.append(
            f"[Documento {i}] {source} (fragmento {chunk_idx+1}/{total}):\n{doc.strip()}"
        )
        citations.append({
            "doc": source,
            "chunk": chunk_idx,
            "score": float(dist)
        })
    
    return "\n\n---\n\n".join(context_parts), citations


def _compute_confidence(distances: List[float]) -> float:
    """Compute a confidence score from ChromaDB distances."""
    if not distances:
        return 0.0
    avg_dist = sum(distances) / len(distances)
    # Normalize: lower distance = higher confidence
    # Typical L2 distances with MiniLM: 0.3 (very close) to 2.0 (far)
    confidence = max(0.0, min(1.0, 1.0 - (avg_dist / 2.5)))
    return round(confidence, 3)


def _build_system_prompt() -> str:
    return (
        "Eres un Auditor Interno Senior de la empresa. Resuelves consultas de empleados "
        "basándote ÚNICAMENTE en la normativa y los informes históricos proporcionados en el contexto. "
        "Sigue estas reglas estrictas:\n"
        "1. Responde en español, con tono profesional, claro y paso a paso.\n"
        "2. Cita explícitamente las fuentes usando el formato [Fuente: nombre_documento].\n"
        "3. Si la información del contexto no es suficiente, indica claramente qué parte no puedes resolver "
        "y sugiere elevar la consulta al equipo de Auditoría.\n"
        "4. NUNCA menciones que eres una inteligencia artificial, un modelo de lenguaje o un bot. "
        "Si te preguntan quién eres, di que eres parte del equipo de Auditoría Interna.\n"
        "5. No inventes datos, montos, normas ni procedimientos que no estén en el contexto.\n"
        "6. Si la consulta implica una irregularidad potencial, sé objetivo y orienta al procedimiento formal.\n"
        "7. Mantén la confidencialidad: no repitas información personal de otros empleados salvo que sea necesaria para la consulta.\n"
    )


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
            temperature=0.3,
            max_tokens=1500
        )
    except Exception as e:
        chat_completion = client.chat.completions.create(
            model=GROQ_FALLBACK_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
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
    except Exception:
        return False
