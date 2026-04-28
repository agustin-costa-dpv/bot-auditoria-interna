# Auditoría Interna — Chatbot de Telegram

Chatbot gratuito para resolver consultas de empleados basándose en **normativa interna** e **informes históricos de auditoría**.  
Stack 100% free: Telegram Bot + FastAPI (Render) + Groq (Llama 3) + ChromaDB + SQLite.

---

## Arquitectura

```
Empleado → Telegram → Webhook (FastAPI/Render)
                              ↓
                    +-------------------+
                    |   ChromaDB (RAG)  | ← embeddings locales
                    |   SQLite (historial + perfiles)
                    +-------------------+
                              ↓
                       Groq API (Llama 3)
                              ↓
                    Respuesta con citas
```

---

## Archivos del proyecto

| Archivo | Función |
|---------|---------|
| `main.py` | Servidor FastAPI + webhook Telegram |
| `bot.py` | Lógica del bot: delays humanos, horario laboral, escalamiento |
| `rag_engine.py` | Motor RAG: búsqueda en ChromaDB + prompt a Groq |
| `embedder.py` | Indexación de PDFs en ChromaDB (`all-MiniLM-L6-v2`) |
| `database.py` | SQLite: usuarios, conversaciones, escalaciones, estadísticas |
| `requirements.txt` | Dependencias |

---

## Requisitos previos

- Cuenta en [Render](https://render.com) (plan gratuito o el de $7/mes)
- Cuenta en [Groq](https://console.groq.com) (API key gratuita)
- Bot de Telegram creado con [@BotFather](https://t.me/BotFather)
- PDFs con normativa e informes de auditoría

---

## Variables de entorno

Copiá `.env.example` a `.env` (o configurá directamente en Render):

```bash
# Telegram (obligatorio)
TELEGRAM_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
WEBHOOK_URL=https://tu-servicio.onrender.com/webhook

# Groq (obligatorio)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Admin opcional (para endpoints /stats, /reindex, /escalations)
ADMIN_TOKEN=un_token_secreto

# Horario laboral (opcional, default Lun-Vie 9-18 hs America/Argentina/Buenos_Aires)
TIMEZONE=America/Argentina/Buenos_Aires
BUSINESS_START=9
BUSINESS_END=18

# Rutas locales (no tocar en Render)
DB_PATH=./data/auditoria.db
CHROMA_PATH=./chroma_db
DOCS_DIR=./data/documents
```

---

## Pasos para deployar en Render

### 1. Subir PDFs

Colocá todos los PDFs de normativa e informes en la carpeta:

```
data/documents/
```

Subí esto al repo de GitHub junto con el código, o subilos luego a Render via Shell.

### 2. Crear servicio en Render

1. En Render, creá un **Web Service** nuevo.
2. Conectá tu repositorio de GitHub.
3. Configurá el **Start Command**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
4. Agregá las **Environment Variables** listadas arriba.
5. Deployá.

### 3. Indexar documentos

Una vez deployado, ejecutá el indexador. Podés hacerlo desde el shell de Render:

```bash
cd /opt/render/project/src
python embedder.py --reset
```

O usá el endpoint protegido (si configuraste `ADMIN_TOKEN`):

```bash
curl -X POST "https://tu-servicio.onrender.com/reindex?admin_token=un_token_secreto&reset=true"
```

### 4. Configurar webhook de Telegram

El servidor intenta setear el webhook automáticamente al iniciar si tenés `WEBHOOK_URL` configurado.  
Verificá que esté activo:

```bash
curl https://api.telegram.org/bot<TELEGRAM_TOKEN>/getWebhookInfo
```

---

## Endpoints útiles

| Endpoint | Uso |
|----------|-----|
| `GET /` | Health check |
| `POST /webhook` | Recibe mensajes de Telegram |
| `GET /stats?admin_token=XXX` | Estadísticas globales |
| `GET /escalations?admin_token=XXX` | Escalamientos pendientes |
| `POST /reindex?admin_token=XXX&reset=true` | Reindexar PDFs |

---

## Características implementadas

- ✅ **Respuestas paso a paso** con citas explícitas a documentos fuente
- ✅ **Delay artificial** de 2.5–5 segundos + indicador "está escribiendo..."
- ✅ **Horario laboral**: solo responde dentro del horario configurado (Lun-Vie)
- ✅ **Escalamiento automático** si la confianza del RAG es < 0.40
- ✅ **Base de datos SQLite** con historial completo, perfiles y estadísticas
- ✅ **Nunca revela** que es un bot; responde como Auditoría Interna
- ✅ **Modelo fallback** si el primero falla en Groq
- ✅ **Shortcuts para saludos** (ahorra tokens del LLM)
- ✅ **Endpoints administrativos** para estadísticas y reindexación

---

## Seguridad y privacidad

- Los datos de chat se almacenan localmente en SQLite (no en servicios externos).
- Groq no retiene ni entrena con los datos enviados (política de Groq Cloud).
- Los embeddings corren 100% localmente (CPU); los documentos nunca salen del servidor.
- El `ADMIN_TOKEN` protege los endpoints sensibles.

---

## Próximos pasos sugeridos

1. Probar con casos reales de empleados.
2. Ajustar `TOP_K` o el chunk size si las respuestas no son precisas.
3. Agregar más modelos o hacer A/B testing con Groq.
4. Implementar notificaciones de escalamiento por email al equipo de Auditoría.

---

## Licencia

Uso interno. Modificá libremente según las necesidades de tu área de Auditoría.
