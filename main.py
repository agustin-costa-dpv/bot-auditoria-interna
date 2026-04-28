"""
FastAPI server for Telegram webhook.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

import bot
import database as db
import embedder

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # https://your-render-service.onrender.com/webhook
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}" if TELEGRAM_TOKEN else None


async def send_telegram_message(chat_id: int, text: str) -> bool:
    """Send a text message via Telegram Bot API. Falls back to plain text if Markdown fails."""
    if not TELEGRAM_API:
        return False
    url = f"{TELEGRAM_API}/sendMessage"
    try:
        async with httpx.AsyncClient() as client:
            # Try Markdown first
            resp = await client.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }, timeout=30.0)
            data = resp.json()
            if data.get("ok"):
                return True
            # If Markdown parse error, retry as plain text
            if "can't parse" in str(data.get("description", "")).lower() or "bad request" in str(data.get("description", "")).lower():
                resp2 = await client.post(url, json={
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True
                }, timeout=30.0)
                return resp2.json().get("ok", False)
            print(f"[API] Telegram error: {data}")
            return False
    except Exception as e:
        print(f"[API] Failed to send message: {e}")
        return False


async def handle_update(update: dict) -> None:
    """Process a single Telegram update."""
    message = update.get("message")
    if not message:
        return

    text = message.get("text")
    if not text:
        # Ignore non-text messages
        return

    chat = message.get("chat", {})
    from_user = message.get("from", {})

    chat_id = chat.get("id")
    username = from_user.get("username")
    first_name = from_user.get("first_name")
    last_name = from_user.get("last_name")

    result = await bot.process_message(
        chat_id=chat_id,
        text=text,
        username=username,
        first_name=first_name,
        last_name=last_name
    )

    if result.get("error"):
        print(f"[Handler] Error for chat {chat_id}: {result['error']}")

    await send_telegram_message(chat_id, result["reply_text"])


async def set_webhook() -> bool:
    """Configure Telegram webhook on startup."""
    if not WEBHOOK_URL or not TELEGRAM_API:
        print("[Startup] WEBHOOK_URL not set, skipping webhook configuration.")
        return False
    url = f"{TELEGRAM_API}/setWebhook"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={
                "url": WEBHOOK_URL,
                "max_connections": 10,
                "allowed_updates": ["message"]
            }, timeout=20.0)
            data = resp.json()
            if data.get("ok"):
                print(f"[Startup] Webhook set: {WEBHOOK_URL}")
                return True
            else:
                print(f"[Startup] Webhook failed: {data}")
                return False
    except Exception as e:
        print(f"[Startup] Webhook error: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    if TELEGRAM_TOKEN and WEBHOOK_URL:
        await set_webhook()
    else:
        print("[Startup] Running without Telegram webhook auto-setup.")

    # Check index health
    if embedder.get_collection_stats()["total_chunks"] == 0:
        print("[Startup] WARNING: ChromaDB collection is empty. Run indexing before using RAG.")
    else:
        print("[Startup] ChromaDB index ready.")

    yield
    # Shutdown
    print("[Shutdown] Cleaning up...")


app = FastAPI(title="Auditoria Interna Bot", lifespan=lifespan)


@app.get("/")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "auditoria-bot"}


@app.post("/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive Telegram webhook updates."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Process in background so Telegram gets immediate 200 OK
    background_tasks.add_task(handle_update, payload)
    return JSONResponse(content={"ok": True})


@app.get("/stats")
async def global_stats(admin_token: Optional[str] = None):
    """Global statistics. Protected by ADMIN_TOKEN if set."""
    if ADMIN_TOKEN and admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return db.get_global_stats()


@app.post("/reindex")
async def force_reindex(admin_token: Optional[str] = None, reset: bool = False):
    """Force document reindexing. Protected by ADMIN_TOKEN."""
    if ADMIN_TOKEN and admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        result = embedder.index_documents(reset=reset)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/escalations")
async def list_escalations(admin_token: Optional[str] = None, status: str = "PENDIENTE"):
    """List escalation tickets. Protected."""
    if ADMIN_TOKEN and admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if status == "PENDIENTE":
        return {"escalations": db.get_pending_escalations()}
    return {"escalations": []}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
