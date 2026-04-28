# main.py
import os
import logging
from fastapi import FastAPI, Request
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

import bot as bot_logic
import database as db

app = FastAPI(title="Auditoría Interna Bot")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI iniciado")

@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
        
        if "message" not in data:
            return {"status": "ok"}
        
        message = data["message"]
        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        username = message["from"].get("username")
        first_name = message["from"].get("first_name")
        last_name = message["from"].get("last_name")
        
        result = await bot_logic.process_message(
            chat_id=chat_id,
            text=text,
            username=username,
            first_name=first_name,
            last_name=last_name
        )
        
        await send_telegram_message(chat_id, result["reply_text"])
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"status": "error"}

async def send_telegram_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={
            "chat_id": chat_id,
            "text": text[:4000],
            "parse_mode": "Markdown"
        }, timeout=30.0)

@app.get("/health")
async def health():
    return {"status": "running"}
