# embedder.py
import os
import re
from pathlib import Path

from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Modelo de embeddings más liviano
MODEL_NAME = "paraphrase-MiniLM-L3-v2"  # 17MB vs 80MB del anterior
CHROMA_PATH = "./chroma_db"
DOCS_PATH = "./documentos"

class DocumentEmbedder:
    def __init__(self):
        print("Cargando modelo de embeddings...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Modelo cargado.")
        
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_PATH
        ))
        self.collection = self.client.get_or_create_collection(
            name="auditoria_interna",
            metadata={"hnsw:space": "cosine"}
        )
    
    def extraer_texto_pdf(self, ruta: str) -> str:
        try:
            reader = PdfReader(ruta)
            texto = ""
            for i, pagina in enumerate(reader.pages):
                if i > 50:  # Limitar a 50 páginas por PDF
                    break
                texto += pagina.extract_text() or ""
            return texto[:50000]  # Limitar a 50k caracteres
        except Exception as e:
            print(f"Error leyendo {ruta}: {e}")
            return ""
    
    def dividir_en_chunks(self, texto: str, tamano: int = 300, solapamiento: int = 50) -> list:
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
        
        return chunks[:100]  # Limitar a 100 chunks por documento
    
    def indexar_documentos(self):
        documentos = list(Path(DOCS_PATH).rglob("*.pdf"))
        print(f"Encontrados {len(documentos)} documentos")
        
        ids_existentes = set(self.collection.get()["ids"]) if self.collection.count() > 0 else set()
        
        for doc in documentos:
            doc_id = str(doc.relative_to(DOCS_PATH)).replace("/", "_").replace(".pdf", "")
            
            if doc_id in ids_existentes:
                print(f"Ya indexado: {doc_id}")
                continue
            
            texto = self.extraer_texto_pdf(str(doc))
            if not texto.strip():
                print(f"Sin texto extraíble: {doc}")
                continue
            
            chunks = self.dividir_en_chunks(texto)
            print(f"Indexando {doc.name}: {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                embedding = self.model.encode(chunk, convert_to_numpy=True).tolist()
                
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "fuente": str(doc.relative_to(DOCS_PATH)),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }]
                )
        
        print("Indexación completada")
    
    def buscar(self, consulta: str, n_resultados: int = 3) -> list:
        embedding = self.model.encode(consulta, convert_to_numpy=True).tolist()
        resultados = self.collection.query(
            query_embeddings=[embedding],
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

# Singleton para reutilizar
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = DocumentEmbedder()
        if _embedder.collection.count() == 0:
            _embedder.indexar_documentos()
    return _embedder
