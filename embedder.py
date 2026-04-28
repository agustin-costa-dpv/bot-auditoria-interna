"""
PDF Embedding & Indexing module.
Reads PDFs from data/documents/, chunks them, and indexes into ChromaDB.
"""

import os
import glob
import hashlib
from typing import List, Dict, Any
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./data/documents")
COLLECTION_NAME = "auditoria"

# Local embedding model (free, runs on CPU, ~80MB)
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_chroma_client() -> chromadb.Client:
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_embedding_function():
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def get_or_create_collection(client: chromadb.Client = None):
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"description": "Normativa e informes de auditoría interna"}
    )


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text page by page from a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i, "text": text.strip()})
    return pages


def split_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 300) -> List[str]:
    """Simple recursive-like chunking by paragraphs with overlap."""
    separators = ["\n\n", "\n", ". ", " "]
    
    def _split(t: str, sep_idx: int) -> List[str]:
        if sep_idx >= len(separators):
            # last resort: character split
            chunks = []
            for i in range(0, len(t), chunk_size - chunk_overlap):
                chunk = t[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks
        
        sep = separators[sep_idx]
        parts = t.split(sep)
        chunks = []
        current = ""
        
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    # part too big, recurse with next separator
                    chunks.extend(_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        
        if current.strip():
            chunks.append(current.strip())
        return chunks
    
    return _split(text, 0)


def index_documents(docs_dir: str = None, reset: bool = False) -> Dict[str, Any]:
    """
    Index all PDFs from docs_dir into ChromaDB.
    If reset=True, deletes existing collection and re-indexes everything.
    """
    docs_dir = docs_dir or DOCS_DIR
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    client = get_chroma_client()
    
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[Embeder] Collection '{COLLECTION_NAME}' deleted for reset.")
        except Exception:
            pass
    
    collection = get_or_create_collection(client)
    
    pdf_files = glob.glob(os.path.join(docs_dir, "**/*.pdf"), recursive=True)
    if not pdf_files:
        return {"indexed": 0, "chunks": 0, "message": "No PDFs found."}
    
    total_chunks = 0
    indexed_files = 0
    
    for pdf_path in sorted(pdf_files):
        rel_path = os.path.relpath(pdf_path, docs_dir)
        print(f"[Embeder] Processing: {rel_path}")
        
        try:
            pages = extract_text_from_pdf(pdf_path)
            full_text = "\n\n".join(p["text"] for p in pages)
            
            if not full_text.strip():
                print(f"[Embeder] Skipping (no text): {rel_path}")
                continue
            
            chunks = split_text(full_text)
            doc_id_base = hashlib.md5(rel_path.encode()).hexdigest()[:12]
            
            ids = []
            documents = []
            metadatas = []
            
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id_base}_chunk_{idx}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "source": rel_path,
                    "page_start": pages[0]["page"] if pages else 1,
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                })
            
            # Upsert in batches to avoid huge single payloads
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                end = i + batch_size
                collection.upsert(
                    ids=ids[i:end],
                    documents=documents[i:end],
                    metadatas=metadatas[i:end]
                )
            
            total_chunks += len(chunks)
            indexed_files += 1
            print(f"[Embeder]  -> {len(chunks)} chunks indexed.")
            
        except Exception as e:
            print(f"[Embeder] ERROR processing {rel_path}: {e}")
    
    return {
        "indexed": indexed_files,
        "chunks": total_chunks,
        "message": f"Indexed {indexed_files} files ({total_chunks} chunks)."
    }


def get_collection_stats() -> Dict[str, Any]:
    collection = get_or_create_collection()
    count = collection.count()
    return {"collection_name": COLLECTION_NAME, "total_chunks": count}


if __name__ == "__main__":
    import sys
    reset_flag = "--reset" in sys.argv
    result = index_documents(reset=reset_flag)
    print(result)
    print(get_collection_stats())
