# embedder.py
import os
import re
from pathlib import Path

from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings

CHROMA_PATH = "./chroma_db"
DOCS_PATH = "./documentos"
COLLECTION_NAME = "auditoria_interna"

_client = None
_collection = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_PATH
        ))
    return _client

def get_or_create_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection

def extraer_texto_pdf(ruta: str) -> str:
    try:
        reader = PdfReader(ruta)
        texto = ""
        for i, pagina in enumerate(reader.pages):
            if i > 30:
                break
            texto += pagina.extract_text() or ""
        return texto[:30000]
    except Exception as e:
        print(f"Error leyendo {ruta}: {e}")
        return ""

def dividir_en_chunks(texto: str, tamano: int = 400, solapamiento: int = 50) -> list:
    oraciones = re.split(r'(?<=[.!?])\s+', texto)
    chunks = []
    chunk_actual = ""
    
    for oracion in oraciones:
        if len(chunk_actual) + len(oracion) < tamano:
            chunk_actual += " " + oracion
        else:
            if chunk_actual:
                chunks.append(chunk_actual.strip())
            chunk_actual = oracion
    
    if chunk_actual:
        chunks.append(chunk_actual.strip())
    
    return chunks[:50]

def indexar_documentos():
    collection = get_or_create_collection()
    documentos = list(Path(DOCS_PATH).rglob("*.pdf"))
    print(f"Encontrados {len(documentos)} documentos")
    
    if collection.count() > 0:
        print(f"Ya hay {collection.count()} documentos indexados")
        return
    
    for doc in documentos:
        doc_id = str(doc.relative_to(DOCS_PATH)).replace("/", "_").replace(".pdf", "")
        texto = extraer_texto_pdf(str(doc))
        
        if not texto.strip():
            continue
        
        chunks = dividir_en_chunks(texto)
        print(f"Indexando {doc.name}: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{
                    "fuente": str(doc.relative_to(DOCS_PATH)),
                    "chunk_index": i
                }]
            )
    
    print("Indexación completada")

def buscar(consulta: str, n_resultados: int = 3) -> list:
    collection = get_or_create_collection()
    resultados = collection.query(
        query_texts=[consulta],
        n_results=n_resultados
    )
    
    documentos = []
    for i, doc in enumerate(resultados["documents"][0]):
        metadatos = resultados["metadatas"][0][i]
        distancia = resultados["distances"][0][i]
        documentos.append({
            "texto": doc,
            "fuente": metadatos["fuente"],
            "relevancia": 1 - distancia
        })
    
    return documentos
