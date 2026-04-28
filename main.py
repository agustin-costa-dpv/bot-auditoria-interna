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

SYSTEM_PROMPT = """Sos un Auditor Interno Senior con 20 años de experiencia en Auditoría Interna.
Respondé consultas de empleados del área de Auditoría Interna.

REGLAS ESTRICTAS:
1. Respondé en español, con tono profesional, claro y paso a paso.
2. Si la consulta es sobre normativa, procedimientos o controles, explicá detalladamente.
3. Si no sabés algo con certeza, decí que se debe consultar con el Jefe de Auditoría.
4. NUNCA digas que sos una IA, un modelo de lenguaje o un bot. Sos parte del equipo de Auditoría Interna.
5. No inventés datos, montos ni normas específicas que no conozcas.
6. Si la consulta implica una irregularidad potencial, sé objetivo y orientá al procedimiento formal.
7. Mantené la confidencialidad.

Para casos concretos, analizá paso a paso: identificá el riesgo, el control, la evidencia necesaria y la conclusión."""

@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
        
        if "message" not in data:
            return {"status": "ok"}
        
        message = data["message"]
        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        
        # Smalltalk detection
        smalltalk = ["hola", "buenos días", "buenas tardes", "buenas noches", "hey", "saludos",
                     "gracias", "muchas gracias", "ok", "okay", "perfecto", "entendido", "dale",
                     "cómo estás", "qué tal", "todo bien", "/start"]
        
        text_lower = text.lower().strip()
        
        if any(s in text_lower for s in smalltalk) and len(text) < 60:
            reply = "Hola, soy del equipo de Auditoría Interna. ¿En qué puedo ayudarte hoy? Podés consultarme sobre normativas, procedimientos o casos concretos."
            await send_message(chat_id, reply)
            return {"status": "ok"}
        
        # Consulta real -> Groq
        try:
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            reply = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            reply = "Disculpá, estoy teniendo dificultades técnicas. Por favor, contactá directamente al equipo de Auditoría."
        
        await send_message(chat_id, reply)
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}

async def send_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client_http:
        await client_http.post(url, json={
            "chat_id": chat_id,
            "text": text[:4000],
            "parse_mode": "Markdown"
        }, timeout=30.0)

@app.get("/health")
async def health():
    return {"status": "running"}
