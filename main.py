# main.py
import os
import logging
from fastapi import FastAPI, Request
import httpx
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
app = FastAPI(title="Auditoría Interna Bot")


def extraer_texto_pdf(ruta, max_paginas=15):
    """Extrae texto de un PDF, limitado a N páginas."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(ruta)
        texto = ""
        for i, pagina in enumerate(reader.pages):
            if i >= max_paginas:
                break
            texto += pagina.extract_text() or ""
        return texto[:8000]
    except Exception as e:
        logger.error(f"Error leyendo {ruta}: {e}")
        return ""


def cargar_documentos():
    """Carga todos los PDFs de documentos/normativa y documentos/informes."""
    documentos = {}
    docs_path = "./documentos"

    if not os.path.exists(docs_path):
        logger.warning(f"No existe carpeta {docs_path}")
        return documentos

    for carpeta in ["normativa", "informes"]:
        carpeta_path = os.path.join(docs_path, carpeta)
        if not os.path.exists(carpeta_path):
            continue

        for archivo in os.listdir(carpeta_path):
            if archivo.endswith(".pdf"):
                ruta = os.path.join(carpeta_path, archivo)
                logger.info(f"Extrayendo: {archivo}")
                texto = extraer_texto_pdf(ruta)
                if texto.strip():
                    documentos[archivo] = texto
                    logger.info(f"  -> {len(texto)} caracteres extraidos")

    logger.info(f"Total documentos cargados: {len(documentos)}")
    return documentos


# Cargar documentos al iniciar
DOCUMENTOS = cargar_documentos()

SYSTEM_PROMPT_BASE = """Sos un Auditor Interno Senior con 20 anos de experiencia.
Respondé consultas de empleados del area de Auditoria Interna.
REGLAS:
1. Respondé en español, tono profesional, claro y paso a paso.
2. BASATE UNICAMENTE en la documentacion proporcionada. NO inventes normas.
3. Citá las fuentes como [Fuente: nombre_del_archivo].
4. Si la documentacion no alcanza, deci que se debe consultar con el Jefe de Auditoria.
5. NUNCA digas que sos una IA. Sos parte del equipo de Auditoria Interna.
6. No inventes datos, montos ni normas especificas que no esten en la documentacion.
7. Para casos concretos, analizá paso a paso: riesgo, control, evidencia, conclusion."""


def construir_prompt(consulta):
    """Arma el prompt completo con la documentacion."""
    if not DOCUMENTOS:
        return SYSTEM_PROMPT_BASE, consulta

    contexto_parts = []
    for nombre, contenido in DOCUMENTOS.items():
        resumen = contenido[:2500]
        contexto_parts.append(f"[Documento: {nombre}]\n{resumen}\n")

    contexto = "\n---\n".join(contexto_parts)
    system_prompt = SYSTEM_PROMPT_BASE + f"\n\nDOCUMENTACION DISPONIBLE:\n\n{contexto}"
    return system_prompt, consulta


@app.post("/webhook")
async def webhook(request: Request):
    """Recibe mensajes de Telegram."""
    try:
        data = await request.json()

        if "message" not in data:
            return {"status": "ok"}

        message = data["message"]
        chat_id = message["chat"]["id"]
        text = message.get("text", "")

        # Detectar smalltalk (saludos, gracias, etc.)
        smalltalk = [
            "hola", "buenos dias", "buenas tardes", "buenas noches", "hey", "saludos",
            "gracias", "muchas gracias", "ok", "okay", "perfecto", "entendido", "dale",
            "como estas", "que tal", "todo bien", "/start"
        ]

        text_lower = text.lower().strip()

        if any(s in text_lower for s in smalltalk) and len(text) < 60:
            reply = "Hola, soy del equipo de Auditoria Interna. ¿En que puedo ayudarte hoy? Podes consultarme sobre normativas, procedimientos o casos concretos."
            await send_message(chat_id, reply)
            return {"status": "ok"}

        # Consulta real con contexto documental
        system_prompt, user_query = construir_prompt(text)

        try:
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Consulta del empleado: {user_query}"}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            reply = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            reply = "Disculpa, estoy teniendo dificultades tecnicas. Por favor, contacta directamente al equipo de Auditoria."

        await send_message(chat_id, reply)
        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}


async def send_message(chat_id: int, text: str):
    """Envia mensaje a Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client_http:
        await client_http.post(url, json={
            "chat_id": chat_id,
            "text": text[:4000],
            "parse_mode": "Markdown"
        }, timeout=30.0)


@app.get("/health")
async def health():
    """Endpoint de salud."""
    return {"status": "running", "docs_loaded": len(DOCUMENTOS)}
