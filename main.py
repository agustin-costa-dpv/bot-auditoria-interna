# main.py
import gc
from fastapi import FastAPI, Request
from telegram import Update

from bot import application
from database import init_db

# Inicializar base de datos al arrancar
init_db()

app = FastAPI(title="Auditoría Interna Bot")

@app.post("/webhook")
async def webhook(request: Request):
    """Recibe actualizaciones de Telegram."""
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"status": "ok"}

@app.get("/health")
async def health():
    """Endpoint de salud para monitoreo."""
    return {"status": "running", "service": "auditoria-interna-bot"}

@app.get("/")
async def root():
    return {
        "message": "Bot de Auditoría Interna activo",
        "endpoints": ["/webhook", "/health"]
    }

@app.post("/clear-memory")
async def clear_memory():
    """Liberar memoria manualmente."""
    gc.collect()
    return {"status": "memory cleared"}
